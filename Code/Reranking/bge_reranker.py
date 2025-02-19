from FlagEmbedding import LayerWiseFlagLLMReranker
import json
import pytrec_eval
from tqdm import tqdm
import re
import os
import subprocess
from itertools import product
import numpy as np
import shutil

# from dense_un import read_dataset_info, read_query_info


def read_dataset_info(filename='dataset_metadata.json'):
    dataset_info_dict = {}
    with open(filename, 'r') as f:
        data = json.load(f)
    for item in data:
        dataset_id = item[0]
        info = '\n'.join(item[1:])
        dataset_info_dict[dataset_id] = info
    return dataset_info_dict


def read_query_info(filename='query_info.json'):
    query_info_dict = {}
    pattern = r' \[SEP\] '
    with open(filename, 'r') as f:
        data = json.load(f)
        for item in data:
            query_info_dict[item['id']] = '\n'.join(re.split(pattern, item['content']))
    return query_info_dict


def output_eval_result(qrels_dict, run_dict, metrics):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, metrics)
    eval_results = evaluator.evaluate(run_dict)
    results = {}
    for metric in metrics:
        results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)
    for metric in metrics:
        print(f'{metric}: {results[metric]:.4f}', end='\t')
    print()


def rerank(model_name, rerank_file, qrels_file, save_path, cutoff_layers=[28]):
    reranker = LayerWiseFlagLLMReranker(model_name, use_fp16=True, trust_remote_code=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation

    dataset_info_dict = read_dataset_info()
    query_info_dict = read_query_info(filename='fold_100/query_info_all.json')

    if isinstance(rerank_file, str):
        with open(rerank_file, 'r') as f:
            rerank_dict = json.load(f)
    elif isinstance(rerank_file, dict):
        rerank_dict = rerank_file
    else:
        raise TypeError("Invalid reranker_file type!", type(rerank_file))
    
    results = {}
    with tqdm(total=len(rerank_dict)) as pbar:
        for query_id, dataset_ids in rerank_dict.items():
            qry_doc = [[query_info_dict[query_id], dataset_info_dict[x]] for x in dataset_ids]
            scores = reranker.compute_score(qry_doc, cutoff_layers=cutoff_layers)
            results[query_id] = {dataset_id: score for dataset_id, score in zip(dataset_ids, scores)}
            pbar.update(1)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    with open(qrels_file, 'r') as f:
        qrels_dict = json.load(f)

    print('='*10 + model_name + '='*10)
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    output_eval_result(qrels_dict, results, metrics)

    return results


def generate_train_data(train_qrels_file, output_file, cache_dir='.cache', hn_range='10-210', candidate_pool=None, merge_valid_qrels_file=None):
    dataset_info_dict = read_dataset_info()
    query_info_dict = read_query_info(filename='fold_100/query_info_all.json')

    with open(train_qrels_file, 'r') as f:
        train_qrels = json.load(f)

    if merge_valid_qrels_file:
        with open(merge_valid_qrels_file, 'r') as f:
            valid_qrels = json.load(f)
            train_qrels.update(valid_qrels)

    os.makedirs(cache_dir, exist_ok=True)
    temp_qrels = os.path.join(cache_dir, 'temp1.jsonl')

    with open(temp_qrels, 'w') as f:
        for query_id, rel_dict in train_qrels.items():
            query_info = query_info_dict[query_id]
            pos_list, neg_list = [], []
            for dataset_id, rel in rel_dict.items():
                dataset_info = dataset_info_dict[dataset_id]
                if rel > 0:
                    pos_list.append(dataset_info)
                else:
                    neg_list.append(dataset_info)
            if len(pos_list) > 0:
                jsonl_data = {
                    'query': query_info,
                    'pos': pos_list,
                    'neg': neg_list,
                }
                json.dump(jsonl_data, f)
                f.write('\n')
    
    # Hard Negatives
    temp_hn = os.path.join(cache_dir, 'temp2.jsonl')
    cmd_hn = [
        "python", "hn_mine.py",
        "--embedder_name_or_path", "BAAI/bge-base-en-v1.5",
        "--input_file", os.path.abspath(temp_qrels),
        "--output_file", os.path.abspath(temp_hn),
        "--range_for_sampling", hn_range,
        "--negative_number", "15"
    ]
    if candidate_pool:
        cmd_hn += ["--candidate_pool", os.path.abspath(candidate_pool)]
    try:
        result = subprocess.run(cmd_hn, cwd="FlagEmbedding/scripts", check=True, text=True, capture_output=True)
        print("Hard negatives successfully.")
    except subprocess.CalledProcessError as e:
        print("HN Command failed with error.")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
    
    # Teacher Scores
    cmd_score = [
        "python", "add_reranker_score.py",
        "--input_file", os.path.abspath(temp_hn),
        "--output_file", os.path.abspath(output_file),
        "--reranker_name_or_path", "BAAI/bge-reranker-v2-minicpm-layerwise",
        "--reranker_query_max_length", "512",
        "--reranker_max_length", "1024"
    ]
    try:
        result = subprocess.run(cmd_score, cwd="FlagEmbedding/scripts", check=True, text=True, capture_output=True)
        print("Teacher Scores successfully.")
    except subprocess.CalledProcessError as e:
        print("TS Command failed with error.")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)


def run_fine_tuning_bash(train_data_file, per_device_train_batch_size, learning_rate, num_train_epochs=6):
    # Set environment variable for WANDB
    os.environ["WANDB_MODE"] = "disabled"

    # Define script variables
    train_data = train_data_file
    num_train_epochs = num_train_epochs
    per_device_train_batch_size = per_device_train_batch_size
    gradient_accumulation_steps = 1
    train_group_size = 8
    learning_rate = learning_rate
    num_gpus = 4

    # Set HF_HUB_CACHE if not set
    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")

    if "HF_HOME" not in os.environ:
        os.environ['HF_HOME'] = os.path.expanduser("~/.cache/huggingface")

    model_args = f""" \
        --model_name_or_path BAAI/bge-reranker-v2-minicpm-layerwise \
        --cache_dir {os.environ['HF_HUB_CACHE']} \
        --use_lora True \
        --lora_rank 32 \
        --lora_alpha 64 \
        --use_flash_attn True \
        --target_modules q_proj k_proj v_proj o_proj \
        --save_merged_lora_model True \
        --model_type decoder \
        --model_type from_finetuned_model \
        --start_layer 8 \
        --head_multi True \
        --head_type simple \
        --trust_remote_code True \
    """

    data_args = f""" \
        --train_data {train_data} \
        --cache_path ~/.cache \
        --train_group_size {train_group_size} \
        --query_max_len 512 \
        --passage_max_len 512 \
        --pad_to_multiple_of 8 \
        --knowledge_distillation True \
        --query_instruction_for_rerank 'A: ' \
        --query_instruction_format '{{}}{{}}' \
        --passage_instruction_for_rerank 'B: ' \
        --passage_instruction_format '{{}}{{}}' \
    """

    # model_output_path = f'./bge_reranker/bge-reranker-v2-minicpm-layerwise_{learning_rate}_{per_device_train_batch_size}'
    model_output_path = f'./bge_reranker/bge-reranker-v2-minicpm-layerwise_grid-search'
    deepseed_file = './bge_reranker/ds_stage0.json'

    training_args = f""" \
        --output_dir {model_output_path} \
        --overwrite_output_dir \
        --learning_rate {learning_rate} \
        --bf16 \
        --num_train_epochs {num_train_epochs} \
        --per_device_train_batch_size {per_device_train_batch_size} \
        --gradient_accumulation_steps {gradient_accumulation_steps} \
        --dataloader_drop_last True \
        --warmup_ratio 0.1 \
        --gradient_checkpointing \
        --weight_decay 0.01 \
        --logging_steps 1 \
        --save_steps 2000 \
        --deepspeed {deepseed_file} \
    """

    cmd = f"torchrun --nproc_per_node {num_gpus} \
        -m FlagEmbedding.finetune.reranker.decoder_only.layerwise \
        {model_args} \
        {data_args} \
        {training_args}"

    print(f"Running command: {cmd}")
    
    # Run the command using subprocess
    subprocess.run(cmd, shell=True, check=True)

    return model_output_path


def grid_search(fold_start, fold_end):
    param_combinations = list(product(*[
        [1e-05, 5e-06], # learning rate 
        [2, 4], # per_device_train_batch_size (lr * 4 = 8, 16) 
    ]))
    param_names = ['learning rate', 'batch size']

    best_params = None
    best_score = -np.inf

    for params in param_combinations:
        learning_rate, batch_size = params
        scores = []
        for fold in range(fold_start, fold_end):
            train_data_file = f'bge_reranker/train/fold_{fold}.jsonl'
            model_output_path = run_fine_tuning_bash(train_data_file, batch_size, learning_rate)
            valid_qrels_file = f'fold_{fold}/valid.json'
            run_dict = rerank(model_name=model_output_path, 
                              rerank_file=valid_qrels_file,
                              qrels_file=valid_qrels_file, 
                              save_path=f'bge_reranker/results/{fold}_{learning_rate}_{batch_size}.json',
                             )
            with open(valid_qrels_file, 'r') as f:
                valid_qrels = json.load(f)
            metric = 'ndcg_cut_10'
            evaluator = pytrec_eval.RelevanceEvaluator(valid_qrels, [metric])
            eval_result = evaluator.evaluate(run_dict)
            scores += [x[metric] for x in eval_result.values()]
            print(f'fold: {fold}, params:{params}, NDCG@10: {np.mean(scores)}')
        
        mean_score = np.mean(scores)
        results = {
            'fold': (fold_start, fold_end),
            'params': params,
            'mean NDCG@10': mean_score,
        }
        with open('bge_reranker/grid_search_metric.jsonl', 'a') as outfile:
            json.dump(results, outfile)
            outfile.write('\n')
        print(results)
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(zip(param_names, params))

    print(f"best params: {best_params}")
    print(f"best avg NDCG@10: {best_score}")
    with open('bge_reranker/grid_search_metric.jsonl', 'a') as outfile:
        results = {
            'fold': (fold_start, fold_end),
            'best params': best_params,
            'best avg NDCG@10': best_score,
        }
        json.dump(results, outfile)
        outfile.write('\n')


def copy_checkpoint(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)


def train_with_best_params_and_eval(fold_start, fold_end, batch_size, learning_rate, num_train_epochs=10):
    with open('rerank_results/rerank_20.json', 'r') as f:
        reranker_dict = json.load(f)

    run_dict_all = {}
    for fold in range(fold_start, fold_end):
        train_data_file = f'bge_reranker/train/fold_{fold}_merged.jsonl'
        model_output_path = run_fine_tuning_bash(train_data_file, batch_size, learning_rate, num_train_epochs)
        copy_checkpoint(model_output_path, f'bge_reranker/checkpoints/fold_{fold}')
        test_qrels_file = f'fold_{fold}/test.json'
        with open(test_qrels_file, 'r') as f:
            test_qrels = json.load(f)
        sub_reranker_dict = {k: v for k, v in reranker_dict.items() if k in test_qrels.keys()}
        run_dict = rerank(model_name=model_output_path, 
                            rerank_file=sub_reranker_dict,
                            qrels_file=test_qrels_file, 
                            save_path=f'bge_reranker/results/rerank_{fold}_{learning_rate}_{batch_size}.json',
                        )
        run_dict_all.update(run_dict)

    with open(f'bge_reranker/results/rerank_{fold_start}-{fold_end}_{learning_rate}_{batch_size}.json', 'w') as f:
        json.dump(run_dict_all, f)

    with open('fold_100/test.json', 'r') as f:
        qrels_dict = json.load(f)

    print('='*10 + f'bge-reranker-v2-minicpm-layerwise (fold {fold_start}-{fold_end})' + '='*10)
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    output_eval_result(qrels_dict, run_dict_all, metrics)


def fine_tuning():
    for i in list(range(5)):  #  + [100]
        print(f'[fold {i}]')
        generate_train_data(
            train_qrels_file=f'fold_{i}/train.json', 
            output_file=f'bge_reranker/train/fold_{i}.jsonl', 
            cache_dir='bge_reranker/.cache',
            hn_range='2-200',  # 60-300
            candidate_pool='bge_reranker/candidate_pool.jsonl'
        )

    for i in list(range(5)):  #  + [100]
        print(f'[fold {i}]')
        generate_train_data(
            train_qrels_file=f'fold_{i}/train.json', 
            output_file=f'bge_reranker/train/fold_{i}_merged.jsonl', 
            cache_dir='bge_reranker/.cache',
            hn_range='2-200',  # 60-300
            candidate_pool='bge_reranker/candidate_pool.jsonl',
            merge_valid_qrels_file=f'fold_{i}/valid.json',
        )

    grid_search(0, 5)
    grid_search(100, 101)

    train_with_best_params_and_eval(fold_start=0, fold_end=5, batch_size=4, learning_rate=1e-05, num_train_epochs=10)
    train_with_best_params_and_eval(fold_start=100, fold_end=101, batch_size=2, learning_rate=1e-05, num_train_epochs=10)


def run_reranker():
    with open('rerank_results/rerank_20.json', 'r') as f:
        reranker_dict = json.load(f)

    # without ft
    rerank(model_name='BAAI/bge-reranker-v2-minicpm-layerwise', rerank_file='rerank_results/rerank_20.json',
           qrels_file='fold_100/test.json', save_path='rerank_results/bge-reranker-v2-minicpm-layerwise.json',
           cutoff_layers=[40])

    # LLM ft
    rerank(model_name='bge_reranker/checkpoints/fold_100', rerank_file='rerank_results/rerank_20.json',
           qrels_file='fold_100/test.json', save_path='rerank_results/bge-reranker-v2-minicpm-layerwise_llm.json',
           cutoff_layers=[40])

    # 5-fold ft
    run_dict_all = {}
    for fold in range(5):
        test_qrels_file = f'fold_{fold}/test.json'
        with open(test_qrels_file, 'r') as f:
            test_qrels = json.load(f)
        sub_reranker_dict = {k: v for k, v in reranker_dict.items() if k in test_qrels.keys()}
        run_dict = rerank(model_name=f'bge_reranker/checkpoints/fold_{fold}', 
                          rerank_file=sub_reranker_dict,
                          qrels_file=test_qrels_file, 
                          save_path=f'bge_reranker/temp_results.json',
                          cutoff_layers=[40]
                        )
        run_dict_all.update(run_dict)

    with open('rerank_results/bge-reranker-v2-minicpm-layerwise_5-fold.json', 'w') as f:
        json.dump(run_dict_all, f, indent=2)

    with open('fold_100/test.json', 'r') as f:
        qrels_dict = json.load(f)

    print('='*10 + 'bge-reranker-v2-minicpm-layerwise (5-fold)' + '='*10)
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    output_eval_result(qrels_dict, run_dict_all, metrics)


if __name__ == "__main__":
    # Set HF_HUB_CACHE if not set
    if "HF_HUB_CACHE" not in os.environ:
        os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface/hub")
    if "HF_HOME" not in os.environ:
        os.environ['HF_HOME'] = os.path.expanduser("~/.cache/huggingface")
    if "TRANSFORMERS_CACHE" not in os.environ:
        os.environ['TRANSFORMERS_CACHE'] = os.path.expanduser("~/.cache/huggingface/transformers")

    fine_tuning()
    run_reranker()
