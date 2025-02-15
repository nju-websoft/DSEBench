from ragatouille import RAGTrainer, RAGPretrainedModel
import os
import json
from argparse import ArgumentParser
from itertools import product
import numpy as np
import io
import re
import sys
import ctypes
import pytrec_eval
from datetime import datetime
import shutil


parser = ArgumentParser()

parser.add_argument('--doc_maxlen', type=int, default=256)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--fold_start', type=int, default=0)
parser.add_argument('--fold_end', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=5e-6)
parser.add_argument('--mode', choices=['grid_search', 'train', 'without_ft'])


args = parser.parse_args()


def copy_checkpoint(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        d = os.path.join(dest_dir, item)
        if os.path.isfile(s):
            shutil.copy2(s, d)


def train(fold, annotated_pairs, full_corpus, learning_rate, batch_size):
    trainer = RAGTrainer(model_name=f"ColBERT_{fold}", pretrained_model_name="colbert-ir/colbertv2.0", language_code="en")
    trainer.prepare_training_data(
        raw_data=annotated_pairs, 
        data_out_path='data/', 
        all_documents=full_corpus, 
        num_new_negatives=10, 
        mine_hard_negatives=True,
        hard_negative_model_size='small',
        pairs_with_labels=True,
        positive_label=1,
        negative_label=0
    )

    model_path = trainer.train(
        batch_size=batch_size,
        nbits=2, # How many bits will the trained model use when compressing indexes
        maxsteps=500000, # Maximum steps hard stop
        use_ib_negatives=True, # Use in-batch negative to calculate loss
        dim=128, # How many dimensions per embedding. 128 is the default and works well.
        learning_rate=learning_rate, # Learning rate, small values ([3e-6,3e-5] work best if the base model is BERT-like, 5e-6 is often the sweet spot)
        doc_maxlen=args.doc_maxlen, # Maximum document length. Because of how ColBERT works, smaller chunks (128-256) work very well.
        use_relu=False, # Disable ReLU -- doesn't improve performance
        warmup_steps="auto", # Defaults to 10%
    )

    time_now = datetime.now().strftime('%Y-%m/%d')
    directory = os.path.join('.ragatouille/colbert/none', time_now)
    folders = os.listdir(directory)
    folders_with_mtime = [(folder, os.path.getmtime(os.path.join(directory, folder))) for folder in folders]
    latest_folder = max(folders_with_mtime, key=lambda x: x[1])[0]
    ckpt_path = os.path.join(directory, latest_folder, 'checkpoints/colbert')

    print(f'fold: {fold}\ncheckpoint: {ckpt_path}')
    with open('train_log.txt', 'a') as output_file:
        output_file.write(f'{fold}\t{ckpt_path}\t{learning_rate}\t{batch_size}\n')

    copy_checkpoint(src_dir=ckpt_path, dest_dir=f'checkpoints/fold_{fold}')
    return ckpt_path



def pipeline(fold, query_info_dict, dataset_info_dict, learning_rate, batch_size, merge_valid=False):
    full_corpus, document_ids = [], []
    for dataset_id, info in dataset_info_dict.items():
        full_corpus.append(info)
        document_ids.append(dataset_id)

    data = {}
    with open(os.path.join(f'fold_{fold}', 'train.json'), 'r') as f:
        train_qrels = json.load(f)
        data.update(train_qrels)
    if merge_valid:
        with open(os.path.join(f'fold_{fold}', 'valid.json'), 'r') as f:
            valid_qrels = json.load(f)
            data.update(valid_qrels)
    
    annotated_pairs = []
    for query_id, rel_dict in data.items():
        query_info = query_info_dict[query_id]
        for dataset_id, rel in rel_dict.items():
            dataset_info = dataset_info_dict[dataset_id]
            rel = 1 if rel > 0 else 0
            annotated_pairs.append([query_info, dataset_info, rel])

    ckpt_path = train(fold=fold, annotated_pairs=annotated_pairs, full_corpus=full_corpus, learning_rate=learning_rate, batch_size=batch_size)

    RAG = RAGPretrainedModel.from_pretrained(ckpt_path)
    RAG.index(
        collection=full_corpus, 
        document_ids=document_ids,
        index_name=f"fold_{fold}", 
        split_documents=False,
        use_faiss=True,
    )
    return RAG


def grid_search():
    query_info_dict = {}
    with open('query_info.json', 'r') as f:
        data = json.load(f)
        for item in data:
            query_info_dict[item['id']] = item['content']
    
    dataset_info_dict = {}
    with open('dataset_info.json', 'r') as f:
        data = json.load(f)
        for item in data:
            dataset_info_dict[item['id']] = item['content']

    param_combinations = list(product(*[
        [1e-5, 5e-6], # learning rate 
        [8, 16], # batch_size
    ]))
    param_names = ['learning rate', 'batch size']

    best_params = None
    best_score = -np.inf

    for params in param_combinations:
        learning_rate, batch_size = params
        scores = []
        for fold in range(args.fold_start, args.fold_end):
            RAG = pipeline(fold, query_info_dict, dataset_info_dict, learning_rate, batch_size, merge_valid=False)

            run_dict = {}
            with open(os.path.join(f'fold_{fold}', 'valid.json'), 'r') as f:
                valid_qrels = json.load(f)
                for query_id, rel_dict in valid_qrels.items():
                    query_info = query_info_dict[query_id]
                    search_result = RAG.search(query=query_info, k=args.top_k, force_fast=True)
                    run_dict[query_id] = {x['document_id']: x['score'] for x in search_result}
            metric = 'ndcg_cut_10'
            evaluator = pytrec_eval.RelevanceEvaluator(valid_qrels, [metric])
            eval_result = evaluator.evaluate(run_dict)
            scores += [x[metric] for x in eval_result.values()]
            print(f'fold: {fold}, params:{params}, NDCG@10: {np.mean(scores)}')
        
        mean_score = np.mean(scores)
        with open('grid_search_metric.txt', 'a') as outfile:
            outfile.write(f'{params}\t{mean_score:4f}\n')
        print(f'params: {params}, mean NDCG@10: {mean_score}')
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(zip(param_names, params))

    print(f"best params: {best_params}")
    print(f"best avg NDCG@10: {best_score}")


def train_with_best_params():
    learning_rate, batch_size = args.learning_rate, args.batch_size
    print(f'learning_rate: {learning_rate}, batch_size: {batch_size}')

    query_info_dict = {}
    with open('fold_100/query_info_all.json', 'r') as f:
        data = json.load(f)
        for item in data:
            query_info_dict[item['id']] = item['content']
    
    dataset_info_dict = {}
    with open('dataset_corpus_info.json', 'r') as f:
        data = json.load(f)
        for item in data:
            dataset_info_dict[item['id']] = item['content']

    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    eval_results = {}
    for fold in range(args.fold_start, args.fold_end):
        RAG = pipeline(fold, query_info_dict, dataset_info_dict, learning_rate, batch_size, merge_valid=True)

        run_dict = {}
        with open(os.path.join(f'fold_{fold}', 'test.json'), 'r') as f:
            test_qrels = json.load(f)
            for query_id, rel_dict in test_qrels.items():
                query_info = query_info_dict[query_id]
                search_result = RAG.search(query=query_info, k=args.top_k, force_fast=True)
                run_dict[query_id] = {x['document_id']: x['score'] for x in search_result}
        
        with open(os.path.join(f'fold_{fold}', 'search_result.json'), 'w') as f:
            json.dump(run_dict, f)
        evaluator = pytrec_eval.RelevanceEvaluator(test_qrels, metrics)
        eval_result = evaluator.evaluate(run_dict)
        eval_results.update(eval_result)
    
    results = {}
    for metric in metrics:
        results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)
    for metric in metrics:
        print(f'{metric}: {results[metric]:.4f}', end='\t')
    print()


def search_without_ft():
    fold = 100

    query_info_dict = {}
    with open('fold_100/query_info_all.json', 'r') as f:
        data = json.load(f)
        for item in data:
            query_info_dict[item['id']] = item['content']
    
    dataset_info_dict = {}
    with open('dataset_corpus_info.json', 'r') as f:
        data = json.load(f)
        for item in data:
            dataset_info_dict[item['id']] = item['content']
    
    full_corpus, document_ids = [], []
    for dataset_id, info in dataset_info_dict.items():
        full_corpus.append(info)
        document_ids.append(dataset_id)

    RAG = RAGPretrainedModel.from_pretrained('colbert-ir/colbertv2.0')
    RAG.index(
        collection=full_corpus, 
        document_ids=document_ids,
        index_name=f"fold_{fold}_without_ft", 
        split_documents=False,
        use_faiss=True,
    )

    run_dict = {}
    with open(os.path.join(f'fold_{fold}', 'test.json'), 'r') as f:
        test_qrels = json.load(f)
        for query_id, rel_dict in test_qrels.items():
            query_info = query_info_dict[query_id]
            search_result = RAG.search(query=query_info, k=args.top_k, force_fast=True)
            run_dict[query_id] = {x['document_id']: x['score'] for x in search_result}
    
    with open(os.path.join(f'results', 'fold_100_ColBERTv2_without_ft.json'), 'w') as f:
        json.dump(run_dict, f)
    
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    eval_results = {}
    evaluator = pytrec_eval.RelevanceEvaluator(test_qrels, metrics)
    eval_result = evaluator.evaluate(run_dict)
    eval_results.update(eval_result)
    
    results = {}
    for metric in metrics:
        results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)
    for metric in metrics:
        print(f'{metric}: {results[metric]:.4f}', end='\t')
    print()


if __name__ == "__main__":
    if args.mode == 'grid_search':
        grid_search()
    elif args.mode == 'train':
        train_with_best_params()
    elif args.mode == 'without_ft':
        search_without_ft()