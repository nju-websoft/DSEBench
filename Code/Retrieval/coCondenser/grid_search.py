from argparse import ArgumentParser
import subprocess
import json
import os
import pytrec_eval
import numpy as np
from itertools import product
import logging
import shutil


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

parser = ArgumentParser()
parser.add_argument('--pipeline', required=True)
parser.add_argument('--model_src', required=True)
parser.add_argument('--model_dest', required=True)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--fold_start', type=int, default=0)
parser.add_argument('--fold_end', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=5e-6)
parser.add_argument('--mode', choices=['grid_search', 'train', 'without_ft'])
args = parser.parse_args()


def grid_search(fold_start=0, fold_end=5):
    data_save_to = './bert'
    encoding_save_to = './encoding'
    merge_valid = 0
    top_k = 10

    param_combinations = list(product(*[
        [5e-6, 1e-5], # learning rate
        [8, 16], # batch_size
    ]))
    param_names = ['learning rate', 'batch size']

    best_params = None
    best_score = -np.inf

    for params in param_combinations:
        learning_rate, batch_size = params
        scores = []
        for fold in range(fold_start, fold_end):
            qrels_dir = f'fold_{fold}'
            result = subprocess.run(['bash', args.pipeline, data_save_to, encoding_save_to, qrels_dir, str(learning_rate), str(batch_size), str(merge_valid), str(top_k)], capture_output=True, text=True)

            with open(os.path.join(qrels_dir, 'valid.json'), 'r') as f:
                qrels = json.load(f)
            
            run_dict = {}
            with open('dev.rank.tsv', 'r') as f:
                for line in f:
                    if line:
                        query_id, dataset_id, score = line.strip().split()
                        if query_id not in run_dict:
                            run_dict[query_id] = {}
                        run_dict[query_id][dataset_id] = float(score)
            metric = 'ndcg_cut_10'
            evaluator = pytrec_eval.RelevanceEvaluator(qrels, [metric])
            eval_result = evaluator.evaluate(run_dict)
            scores += [x[metric] for x in eval_result.values()]

        mean_score = np.mean(scores)
        with open('grid_search_metric.txt', 'a') as outfile:
            outfile.write(f'{params}\t{mean_score:4f}\n')
        print(f'params: {params}, mean NDCG@10: {np.mean(scores)}')
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(zip(param_names, params))
    
    print(f"best params: {best_params}")
    print(f"best avg NDCG@10: {best_score}")


def train(learning_rate, batch_size, fold_start=0, fold_end=5):
    data_save_to = './bert'
    encoding_save_to = './encoding'
    merge_valid = 1
    top_k = 20

    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    eval_results = {}
    for fold in range(fold_start, fold_end):
        qrels_dir = f'fold_{fold}'
        result = subprocess.run(['bash', args.pipeline, data_save_to, encoding_save_to, qrels_dir, str(learning_rate), str(batch_size), str(merge_valid), str(top_k)], capture_output=True, text=True)
        # print(result)

        # save checkpoint
        src_dir = args.model_src
        dest_dir = os.path.join(args.model_dest, f'fold_{fold}')
        os.makedirs(dest_dir, exist_ok=True)

        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isfile(s):
                shutil.copy2(s, d)

        with open(os.path.join(qrels_dir, 'test.json'), 'r') as f:
            qrels = json.load(f)
        
        run_dict = {}
        with open('dev.rank.tsv', 'r') as f:
            for line in f:
                if line:
                    query_id, dataset_id, score = line.strip().split()
                    if query_id not in run_dict:
                        run_dict[query_id] = {}
                    run_dict[query_id][dataset_id] = float(score)

        with open(os.path.join(qrels_dir, 'search_result.json'), 'w') as f:
            json.dump(run_dict, f)
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
        eval_result = evaluator.evaluate(run_dict)
        eval_results.update(eval_result)
    
    results = {}
    for metric in metrics:
        results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)
    for metric in metrics:
        print(f'{metric}: {results[metric]:.4f}', end='\t')
    print()


def search_without_fine_tuning():
    data_save_to = './bert'
    qrels_dir = 'fold_100'
    encoding_save_to = './encoding'
    if 'dpr' in args.pipeline:
        model_name = '../tevatron/tevatron/examples/dpr/model_nq'
        save_file_name = qrels_dir + '_dpr_without_ft.json'
    else:
        model_name = 'Luyu/co-condenser-marco-retriever'
        save_file_name = qrels_dir + '_coCondenser_without_ft.json'

    top_k = 20

    result = subprocess.run(['bash', 'encode_and_index_search_without_ft.sh', data_save_to, qrels_dir, encoding_save_to, model_name, str(top_k)], capture_output=True, text=True)

    with open(os.path.join(qrels_dir, 'test.json'), 'r') as f:
        qrels = json.load(f)
    run_dict = {}

    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    eval_results = {}

    with open('test.rank.tsv', 'r') as f:
        for line in f:
            if line:
                query_id, dataset_id, score = line.strip().split()
                if query_id not in run_dict:
                    run_dict[query_id] = {}
                run_dict[query_id][dataset_id] = float(score)
    with open(os.path.join('results', save_file_name), 'w') as f:
        json.dump(run_dict, f)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    eval_result = evaluator.evaluate(run_dict)
    eval_results.update(eval_result)

    results = {}
    for metric in metrics:
        results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)
    for metric in metrics:
        print(f'{metric}: {results[metric]:.4f}', end='\t')
    print()


if __name__ == "__main__":
    fold_start, fold_end = args.fold_start, args.fold_end

    if args.mode == 'grid_search':
        print(f'grid_search {fold_start}-{fold_end}')
        grid_search(fold_start, fold_end)
    elif args.mode == 'train':
        learning_rate, batch_size = args.learning_rate, args.batch_size
        print(f'train {fold_start}-{fold_end}: {learning_rate}, {batch_size}')
        train(learning_rate, batch_size, fold_start, fold_end)
    elif args.mode == 'without_ft':
        search_without_fine_tuning()
    
