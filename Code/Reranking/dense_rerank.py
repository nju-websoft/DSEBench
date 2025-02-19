import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json
import re
from sparse import read_qrels_origin, get_qrels_multi, output_eval_result
from unsupervised_dense import read_dataset_info, read_query_info
from sentence_transformers.util import cos_sim
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from torch.nn import DataParallel


def encode_texts(model, texts, prompt_name=None):
    if prompt_name:
        embeddings = model.encode(texts, normalize_embeddings=True, prompt_name=prompt_name)
    else:
        embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings


def rerank(model_name, save_path, top_k=20):
    dataset_info_dict = read_dataset_info()
    query_info_dict = read_query_info()

    model = SentenceTransformer(model_name, trust_remote_code=True).cuda()

    with open('rerank_results/rerank_20.json', 'r') as f:
        rerank_dict = json.load(f)
    
    results = {}
    for query_id, dataset_ids in rerank_dict.items():
        query = query_info_dict[query_id]
        corpus = [dataset_info_dict[x] for x in dataset_ids]

        query_embeddings = encode_texts(model, [query], prompt_name="s2p_query")
        document_embeddings = encode_texts(model, corpus)

        d = document_embeddings.shape[1]
        index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(document_embeddings)

        distances, indices = index.search(query_embeddings, top_k)

        results[query_id] = {}
        i = 0
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


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_embeddings(input_texts, tokenizer, model, device, max_length=4096):
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    
    # Move the tensors to the same device as the model
    batch_dict = {key: tensor.to(device) for key, tensor in batch_dict.items()}
    model.to(device)
    
    with torch.no_grad():
        outputs = model(**batch_dict)
    
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def rerank_7B(model_name, save_path):
    dataset_info_dict = read_dataset_info()
    query_info_dict = read_query_info()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    if torch.cuda.is_available():
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cpu")]

    model.half()
    
    num_devices = len(devices)
    blocks = list(model.children())
    num_blocks = len(blocks)
    blocks_per_device = num_blocks // num_devices

    for i, device in enumerate(devices):
        start = i * blocks_per_device
        end = (i + 1) * blocks_per_device if i != num_devices - 1 else num_blocks
        for block in blocks[start:end]:
            block.to(device)

    with open('rerank_results/rerank_20.json', 'r') as f:
        rerank_dict = json.load(f)

    task = 'Given a web search query, retrieve relevant passages that answer the query'

    results = {}
    with tqdm(total=len(rerank_dict), desc="Processing") as pbar:
        for query_id, dataset_ids in rerank_dict.items():
            pbar.set_postfix(query=query_id, dataset_len=len(dataset_ids))
            query = query_info_dict[query_id]
            query_embedding = get_embeddings([get_detailed_instruct(task, query)], tokenizer, model, devices[0])
            results[query_id] = {}
            for dataset_id in dataset_ids:
                document = dataset_info_dict[dataset_id]
                document_embedding = get_embeddings([document], tokenizer, model, devices[0])
                scores = query_embedding @ document_embedding.T
                score = scores.tolist()[0][0]
                results[query_id][dataset_id] = float(score)
                torch.cuda.empty_cache()
            pbar.update(1)
    
    with open(save_path, 'w') as f:
        json.dump(results, f)

    print('='*10 + model_name + '='*10)
    qrels_kw, qrels_ds = read_qrels_origin()
    qrels_dict = get_qrels_multi(qrels_kw, qrels_ds)
    metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
    output_eval_result(qrels_dict, results, metrics)


if __name__ == "__main__":
    rerank(model_name='dunzhang/stella_en_1.5B_v5', save_path='rerank_results/stella_rerank_results.json', top_k=20)
    rerank_7B(model_name='SFR-Embedding-Mistral', save_path='rerank_results/sfr_rerank_results.json')
