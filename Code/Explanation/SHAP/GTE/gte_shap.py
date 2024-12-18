import argparse

import numpy as np
import os
import pickle
import torch
import shap
import json
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
# from sentence_transformers.util import cos_sim, semantic_search
from utils import get_query_dataset_by_id, dataset_info_from_fields, get_rel_info_by_id
from utils import get_pair_info, get_pooling_result, cosine_similarity, get_pair_id_by_id
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda")


def gte_encoder(sentences, model_path):
    model = SentenceTransformer(model_path, device='cuda')
    embeddings = torch.tensor(model.encode(sentences, device='cuda'))
    return embeddings


def gte_shap_explainer(model_path, pooling_path, annotation_file, dataset_info_path, queries_path, _id, mode):
    query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id)
    pair_id = get_pair_id_by_id(annotation_file, _id)
    with open(pooling_path, 'r') as f:
        data = json.load(f)
    gte_pooling = data[str(pair_id)]
    if candidate_id not in gte_pooling.keys():
        return None
    pair_info, input_dataset_info, candidate_dataset_info = get_pair_info(dataset_info_path, query, input_id, candidate_id, mode)
    ori_query = pair_info
    q_embed = gte_encoder([ori_query.strip()], model_path)[0]
    top_score = gte_pooling[list(gte_pooling.keys())[0]]
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    def gte_predict(texts): #input: a list of string   output: a list of probability [[0.6,0.4] [0.7,0.2]]
        labels = []
        new_texts = []
        for text in texts:
            text = text.replace('[MASK]', ' ')
            new_dataset_info = dataset_info_from_fields(text, candidate_dataset_info)
            new_texts.append(new_dataset_info)
        embeddings = gte_encoder(new_texts, model_path)
        for embedding in embeddings:
            score = cosine_similarity(embedding.cpu(), q_embed.cpu())
            p_rel = min(1 - (top_score - score)/top_score, 1)
            labels.append(p_rel)

        return np.array(labels)

    explainer = shap.Explainer(gte_predict, tokenizer)

    test_text = ["title description tags author summary"]
    exp = explainer(test_text)
    return exp


def gte_shap(model_path, output_file, rel_mode, pooling_path, annotation_file, dataset_info_path, queries_path):
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    tol_num = len(data)
    tol_num = 200
    shap_qrel_results = []
    for _id in tqdm(range(1, tol_num+1)):
        rel_info = get_rel_info_by_id(annotation_file, _id-1)
        query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id-1)
        ind = 500
        if rel_mode == 'qexp':
            ind = 0
        elif rel_mode == 'dexp':
            ind = 2
        if rel_info[ind] != '0':
            shap_result = gte_shap_explainer(model_path, pooling_path, annotation_file, dataset_info_path, queries_path, _id-1, mode)
            if shap_result is None:
                continue
            shap_values = list(shap_result.values[0][1:-1])
            explain = {}
            feature = ['title', 'description', 'tags', 'author', 'summary']
            for i, val in enumerate(shap_values):
                explain[feature[i]] = val
            shap_qrel_results.append({'id': _id, "query": query, "input_id": input_id,
                                      'candidate_id': candidate_id, 'explanation': explain})

    with open(output_file, 'w') as f:
        json.dump(shap_qrel_results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="model_path: model path; mode: 'qd' or 'd' or 'q'; rel: 'qexp' or 'dexp'; output_dir: directory to output file; pooling_path: retrieved file path; "
                                                 "annotation: annotation file path; dataset_info: dataset info file path; queries: queries file path")
    parser.add_argument('--model_path', help="model path", required=True)
    parser.add_argument('--mode', help="'qd' or 'd' or 'q'", required=True)
    parser.add_argument('--rel', help="'qexp' or 'dexp'", required=True)
    parser.add_argument('--output_dir', help="directory to output file", required=True)
    parser.add_argument('--pooling_path', help="retrieved file path", required=True)
    parser.add_argument('--annotation', help="annotation file path", required=True)
    parser.add_argument('--dataset_info', help="dataset info file path", required=True)
    parser.add_argument('--queries', help="queries file path", required=True)
    args = parser.parse_args()
    model_path = args.model_path
    mode = args.mode
    rel = args.rel
    output_dir = args.output_dir
    outfile = os.path.join(output_dir, f'gte_shap_exp_{mode}_{rel}.json')
    pooling = args.pooling_path
    annotation = args.annotation
    dataset_info = args.dataset_info
    queries = args.queries
    gte_shap(model_path, outfile, rel, pooling, annotation, dataset_info, queries)
