import os
from tqdm import tqdm
import argparse
import sklearn
import numpy as np
import copy
import sklearn.ensemble
import sklearn.metrics
import transformers
import shap
from transformers import AutoTokenizer
import json
from rank_bm25 import BM25Okapi
from utils import get_query_dataset_by_id, dataset_info_from_fields, get_rel_info_by_id
from utils import get_pair_info, tokenized_query, get_pair_id_by_id

def get_doc_freq(tokenized_doc):
    frequencies = {}
    for word in tokenized_doc:
        if word not in frequencies:
            frequencies[word] = 0
        frequencies[word] += 1
    return frequencies


def get_corpus(dataset_info_path):
    corpus, tokenized_corpus, ids = [], [], []
    with open(dataset_info_path, 'r') as f:
        dataset_json_data = json.load(f)
    for dataset in dataset_json_data:
        ids.append(dataset['id'])
        content = "\n".join([dataset['title'], dataset['description'], ",".join(dataset['tags']), dataset['author'], dataset['summary']])
        corpus.append(content)
        tokenized_corpus.append(content.split())
    return corpus, tokenized_corpus, ids



def bm25_shap_explainer(_id, ori_bm25, pooling_path, annotation_file, dataset_info_path, queries_path):
    query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id)

    pair_id = get_pair_id_by_id(annotation_file, _id)
    with open(pooling_path, 'r') as f:
        data = json.load(f)
    bm25_pooling = data[str(pair_id)]
    if candidate_id not in bm25_pooling.keys():
        return None
    pair_info, input_dataset_info, candidate_dataset_info = get_pair_info(dataset_info_path, query, input_id, candidate_id, mode)
    ori_query = tokenized_query(pair_info)
    doc_scores = ori_bm25.get_scores(ori_query)
    top_score = max(doc_scores)
    tokenizer = AutoTokenizer.from_pretrained('../../../../model/bert')
    # tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    def bm25_predict(texts): #input: a list of string   output: a list of probability [[0.6,0.4] [0.7,0.2]]
        labels = []

        for text in texts:
            text = text.replace('[MASK]', ' ')
            if len(text.strip()) == 0:
                labels.append(0)
                # labels.append([0.0, 1.0])
                continue
            new_dataset_info = dataset_info_from_fields(text, candidate_dataset_info)
            tokenized_new_dataset_info = new_dataset_info.split()
            ori_bm25.doc_freqs.append(get_doc_freq(tokenized_new_dataset_info))
            doc_len = len(tokenized_new_dataset_info)
            score = 0
            for q in ori_query:
                doc = ori_bm25.doc_freqs[-1]
                q_freq = np.array([doc.get(q) or 0])
                score += (ori_bm25.idf.get(q) or 0) * (q_freq[-1] * (ori_bm25.k1 + 1) /
                                                   (q_freq[-1] + ori_bm25.k1 * (1 - ori_bm25.b + ori_bm25.b * doc_len / ori_bm25.avgdl)))
            ori_bm25.doc_freqs.pop()
            p_rel = min(1 - (top_score - score)/top_score, 1)
            # labels.append([p_rel, 1-p_rel])
            labels.append(p_rel)

        return np.array(labels)

    explainer = shap.Explainer(bm25_predict, tokenizer)

    test_text = ["title description tags author summary"]
    exp = explainer(test_text)
    return exp


def bm25_shap(output_file, rel_mode, pooling_path, annotation_file, dataset_info_path, queries_path):
    ori_corpus, ori_tokenized_corpus, doc_ids = get_corpus(dataset_info_path)
    ori_bm25 = BM25Okapi(ori_tokenized_corpus)
    shap_qrel_results = []
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    tol_num = len(data)
    for _id in tqdm(range(1, tol_num+1)):
        rel_info = get_rel_info_by_id(annotation_file, _id-1)
        query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id-1)
        ind = 500
        if rel_mode == 'qexp':
            ind = 0
        elif rel_mode == 'dexp':
            ind = 2
        if rel_info[ind] != '0':
            shap_result = bm25_shap_explainer(_id-1, ori_bm25, pooling_path, annotation_file, dataset_info_path, queries_path)
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
    parser = argparse.ArgumentParser(description="mode: 'qd' or 'd' or 'q'; rel: 'qexp' or 'dexp'; output_dir: directory to output file; pooling_path: retrieved file path; "
                                                 "annotation: annotation file path; dataset_info: dataset info file path; queries: queries file path")
    parser.add_argument('--mode', help="'qd' or 'd' or 'q'", required=True)
    parser.add_argument('--rel', help="'qexp' or 'dexp'", required=True)
    parser.add_argument('--output_dir', help="directory to output file", required=True)
    parser.add_argument('--pooling_path', help="retrieved file path", required=True)
    parser.add_argument('--annotation', help="annotation file path", required=True)
    parser.add_argument('--dataset_info', help="dataset info file path", required=True)
    parser.add_argument('--queries', help="queries file path", required=True)
    args = parser.parse_args()
    mode = args.mode
    rel = args.rel
    output_dir = args.output_dir
    outfile = os.path.join(output_dir, f'bm25_shap_exp_{mode}_{rel}.json')
    pooling = args.pooling_path
    annotation = args.annotation
    dataset_info = args.dataset_info
    queries = args.queries
    bm25_shap(outfile, rel, pooling, annotation, dataset_info, queries)


