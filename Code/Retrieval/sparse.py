from rank_bm25 import BM25Okapi
import os
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytrec_eval


def read_dataset_metadata(filename='dataset_metadata.json'):
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
    pattern = r' \[SEP\] | '
    with open(filename, 'r') as f:
        data = json.load(f)
        for item in data:
            query_info_dict[item['id']] = '\n'.join(re.split(pattern, item['content']))
    return query_info_dict


def read_qrels_origin(filename='qrels.txt'):
    qrels_kw, qrels_ds = {}, {}
    with open(filename, 'r') as f:
        for line in f:
            if not line:
                continue
            data = line.split('\t')
            query_id, dataset_id, rel_k, rel_d = data[0], data[3], data[4], data[5]
            if query_id not in qrels_kw:
                qrels_kw[query_id] = {}
            qrels_kw[query_id][dataset_id] = int(rel_k)
            if query_id not in qrels_ds:
                qrels_ds[query_id] = {}
            qrels_ds[query_id][dataset_id] = int(rel_d)
    return qrels_kw, qrels_ds


def get_qrels_multi(qrels_kw, qrels_ds):
    qrels_multi = {}
    for query_id, rel_dict in qrels_kw.items():
        qrels_multi[query_id] = {}
        for dataset_id, rel in rel_dict.items():
            qrels_multi[query_id][dataset_id] = int(rel) * int(qrels_ds[query_id][dataset_id])
    return qrels_multi


def output_eval_result(qrels_dict, run_dict, metrics):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, metrics)
    eval_results = evaluator.evaluate(run_dict)
    results = {}
    for metric in metrics:
        results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)
    for metric in metrics:
        print(f'{metric}: {results[metric]:.4f}', end='\t')
    print()


def bm25_search(top_k=20, save_path='bm25_results.json'):
    dataset_info_dict = read_dataset_metadata()

    dataset_ids, corpus = dataset_info_dict.keys(), dataset_info_dict.values()
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query_info_dict = read_query_info()

    results = {}
    for query_id, query_content in query_info_dict.items():
        tokenized_query = query_content.split()
        doc_scores = bm25.get_scores(tokenized_query)
        combined_list = list(zip(doc_scores, dataset_ids))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
        results[query_id] = {x[1]: x[0] for x in sorted_combined_list[:top_k]}
    
    with open(save_path, 'w') as f:
        json.dump(results, f)

    print('='*10 + ' BM25 ' + '='*10)
    qrels_kw, qrels_ds = read_qrels_origin()
    qrels_dict = get_qrels_multi(qrels_kw, qrels_ds)
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    output_eval_result(qrels_dict, results, metrics)


def tfidf_search(top_k=20, save_path='tfidf_results.json'):
    dataset_info_dict = read_dataset_metadata()
    vectorizer = TfidfVectorizer()
    dataset_ids, corpus = list(dataset_info_dict.keys()), list(dataset_info_dict.values())
    dataset_vectors = vectorizer.fit_transform(corpus)

    query_info_dict = read_query_info()
    query_ids, query_content = list(query_info_dict.keys()), list(query_info_dict.values())
    query_vectors = vectorizer.transform(query_content)

    scores = cosine_similarity(query_vectors, dataset_vectors)
    results = {}
    for index, query_id in enumerate(query_ids):
        doc_scores = scores[index]
        combined_list = list(zip(doc_scores, dataset_ids))
        sorted_combined_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
        results[query_id] = {x[1]: x[0] for x in sorted_combined_list[:top_k]}

    with open(save_path, 'w') as f:
        json.dump(results, f)

    print('='*10 + ' TF-IDF ' + '='*10)
    qrels_kw, qrels_ds = read_qrels_origin()
    qrels_dict = get_qrels_multi(qrels_kw, qrels_ds)
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    output_eval_result(qrels_dict, results, metrics)


if __name__ == "__main__":
    # bm25_search()
    tfidf_search()
