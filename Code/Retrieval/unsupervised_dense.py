import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import re
from sparse import read_qrels_origin, get_qrels_multi, output_eval_result
from sentence_transformers.util import cos_sim


def read_dataset_info(filename='dataset_corpus_info.json'):
    dataset_info_dict = {}
    with open(filename, 'r') as f:
        data = json.load(f)
        for item in data:
            dataset_info_dict[item['id']] = item['content']
    return dataset_info_dict


def read_query_info(filename='query_info.json'):
    query_info_dict = {}
    with open(filename, 'r') as f:
        data = json.load(f)
        for item in data:
            query_info_dict[item['id']] = item['content']
    return query_info_dict


def encode_texts(model, texts, prompt_name=None):
    if prompt_name:
        embeddings = model.encode(texts, normalize_embeddings=True, prompt_name=prompt_name)
    else:
        embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


def unsupervised_dense_search(model_name, save_path, top_k=20):
    dataset_info_dict = read_dataset_info()
    query_info_dict = read_query_info()
    dataset_ids, corpus = list(dataset_info_dict.keys()), list(dataset_info_dict.values())
    query_ids, query_content = list(query_info_dict.keys()), list(query_info_dict.values())

    if 'stella_en_1.5B_v5' in model_name:
        model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
        query_embeddings = encode_texts(model, query_content, prompt_name="s2p_query")
        document_embeddings = encode_texts(model, corpus)
    elif 'thenlper/gte-large' in model_name:
        model = SentenceTransformer(model_name)
        query_embeddings = encode_texts(model, query_content)
        document_embeddings = encode_texts(model, corpus)
    elif 'bge-large-en-v1.5' in model_name:
        model = SentenceTransformer(model_name)
        instruction = "Represent this sentence for searching relevant passages:"
        query_embeddings = encode_texts(model, [instruction+q for q in query_content])
        document_embeddings = encode_texts(model, corpus)

    d = document_embeddings.shape[1]
    index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.train(document_embeddings)
    index.add(document_embeddings)

    distances, indices = index.search(query_embeddings, top_k)

    # print(distances)

    results = {}
    for i, query_id in enumerate(query_ids):
        results[query_id] = {}
        for j in range(top_k):
            dataset_id = dataset_ids[indices[i, j]]
            score = distances[i, j]
            results[query_id][dataset_id] = float(score)
    
    with open(save_path, 'w') as f:
        json.dump(results, f)

    print('='*10 + model_name + '='*10)
    qrels_kw, qrels_ds = read_qrels_origin()
    qrels_dict = get_qrels_multi(qrels_kw, qrels_ds)
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    output_eval_result(qrels_dict, results, metrics)


if __name__ == "__main__":
    unsupervised_dense_search(model_name='BAAI/bge-large-en-v1.5', save_path='bge_results.json', top_k=20)
    unsupervised_dense_search(model_name='thenlper/gte-large', save_path='gte_results.json', top_k=20)
    # unsupervised_dense_search(model_name='dunzhang/stella_en_1.5B_v5', save_path='rerank_stella_100.json', top_k=100)
