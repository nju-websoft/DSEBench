from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
import subprocess
from transformers import AutoTokenizer
import os
import json
import argparse
import faiss
import numpy as np
from tqdm import tqdm
from lime.lime_text import LimeTextExplainer
from utils import get_query_dataset_by_id, dataset_info_from_fields, get_rel_info_by_id, get_pair_info, get_pair_id_by_id

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def get_scores(model_type, ckpt_path, query_dict, document_dict, k):
    results = {}
    if model_type == 'ColBERTv2':
        print(ckpt_path)
        RAG = RAGPretrainedModel.from_pretrained(ckpt_path)
        RAG.index(
            collection=list(document_dict.values()),
            document_ids=list(document_dict.keys()),
            index_name="temp_index",
            split_documents=False,
            use_faiss=True,
        )
        for query_id, query_text in query_dict.items():
            search_result = RAG.search(query=query_text, k=k)
            results[query_id] = {x['document_id']: x['score'] for x in search_result}
    elif model_type == 'coCondenser' or model_type == 'DPR':
        tokenizer = AutoTokenizer.from_pretrained('/data1/PTLM/bert-base-uncased/', use_fast=True)
        query_dict_encoded = {k: tokenizer.encode(v, add_special_tokens=False, max_length=512, truncation=True) for k, v
                              in query_dict.items()}
        document_dict_encoded = {k: tokenizer.encode(v, add_special_tokens=False, max_length=512, truncation=True) for
                                 k, v in document_dict.items()}
        # build corpus
        corpus_path = 'bert/corpus'
        os.makedirs(corpus_path, exist_ok=True)
        f = open(os.path.join(corpus_path, 'split00.json'), 'w')
        for document_id, document_encoded in document_dict_encoded.items():
            encoded = {
                'text_id': document_id,
                'text': document_encoded
            }
            f.write(json.dumps(encoded) + '\n')
        f.close()
        # build query
        query_path = 'bert/query'
        os.makedirs(query_path, exist_ok=True)
        f = open(os.path.join(query_path, 'query.json'), 'w')
        for query_id, query_encoded in query_dict_encoded.items():
            encoded = {
                'text_id': query_id,
                'text': query_encoded
            }
            f.write(json.dumps(encoded) + '\n')
        f.close()
        # encode corpus
        encoding_path = 'encoding'
        os.makedirs(os.path.join(encoding_path, 'corpus'), exist_ok=True)
        os.makedirs(os.path.join(encoding_path, 'query'), exist_ok=True)
        command = [
            "python", "-m", "tevatron.driver.encode",
            "--output_dir", "./retriever_model",
            "--model_name_or_path", ckpt_path,
            "--fp16",
            "--p_max_len", "512",
            "--per_device_eval_batch_size", "128",
            "--encode_in_path", os.path.join(corpus_path, 'split00.json'),
            "--encoded_save_path", os.path.join(encoding_path, 'corpus', 'split00.json')
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        # print(result)
        # encode query
        command = [
            "python", "-m", "tevatron.driver.encode",
            "--output_dir", "./retriever_model",
            "--model_name_or_path", ckpt_path,
            "--fp16",
            "--q_max_len", "512",
            "--encode_is_qry",
            "--per_device_eval_batch_size", "128",
            "--encode_in_path", os.path.join(query_path, 'query.json'),
            "--encoded_save_path", os.path.join(encoding_path, 'query', 'qry.pt')
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        # index search
        command = [
            "python", "-m", "tevatron.faiss_retriever",
            "--query_reps", os.path.join(encoding_path, 'query', 'qry.pt'),
            "--passage_reps", os.path.join(encoding_path, 'corpus', 'split00.json'),
            "--depth", str(k),
            "--batch_size", "-1",
            "--save_text",
            "--save_ranking_to", f'{model_type}.tsv'
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        with open(f'{model_type}.tsv', 'r') as f:
            for line in f:
                if line:
                    query_id, document_id, score = line.strip().split()
                    if query_id not in results:
                        results[query_id] = {}
                    results[query_id][document_id] = float(score)
    elif model_type == 'bge' or model_type == 'gte':
        if model_type == 'bge':
            model_name = 'BAAI/bge-large-en-v1.5'
            model = SentenceTransformer(model_name)
            instruction = "Represent this sentence for searching relevant passages:"
            q_embeddings = model.encode([instruction + q for q in query_dict.values()], normalize_embeddings=True)
        elif model_type == 'gte':
            model_name = 'thenlper/gte-large'
            model = SentenceTransformer(model_name)
            q_embeddings = model.encode(list(query_dict.values()), normalize_embeddings=True)
        p_embeddings = model.encode(list(document_dict.values()), normalize_embeddings=True)
        # scores = q_embeddings @ p_embeddings.T
        d = p_embeddings.shape[1]
        index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(p_embeddings)
        distances, indices = index.search(q_embeddings, k)
        query_ids, document_ids = list(query_dict.keys()), list(document_dict.keys())
        for i, query_id in enumerate(query_ids):
            results[query_id] = {}
            for j in range(k):
                document_id = document_ids[indices[i, j]]
                score = distances[i, j]
                results[query_id][document_id] = float(score)
    return results


def dense_lime_explainer(model_type, model_path, retrieved_results, annotation_file, dataset_info_path, queries_path, _id, mode):
    query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id)
    # print(query, input_id, candidate_id)
    dense_pooling = list(retrieved_results.keys())
    # print(dense_pooling)
    if candidate_id not in dense_pooling:
        return None
    pair_info, input_dataset_info, candidate_dataset_info = get_pair_info(dataset_info_path, query, input_id, candidate_id, mode)
    print(pair_info)
    queries = [pair_info]
    query_dict = {f'qry_{i}': q for i, q in enumerate(queries)}
    # print(query_dict)
    top_score = retrieved_results[dense_pooling[0]]
    def dense_predict(texts): #input: a list of string   output: a list of probability [[0.6,0.4] [0.7,0.2]]
        labels = [[0, 1]]*len(texts)
        new_texts = []
        for i, text in enumerate(texts):
            new_dataset_info = dataset_info_from_fields(text, candidate_dataset_info)
            print(text)
            print(new_dataset_info)
            new_texts.append(new_dataset_info)
        # print(len(new_texts))
        document_dict = {f'doc_{i}': d for i, d in enumerate(new_texts)}
        results = get_scores(model_type=model_type,
                             ckpt_path=model_path,
                             query_dict=query_dict,
                             document_dict=document_dict,
                             k=50)
        for k, v in results['qry_0'].items():
            score = v
            p_rel = min(1 - (top_score - score)/top_score, 1)
            # print(score, top_score)
            # print(p_rel, 1-p_rel)
            labels[eval(k.strip('doc_'))] = [p_rel, 1-p_rel]

        return np.array(labels)

    explainer = LimeTextExplainer(class_names=['relevant', 'irrelevant'])

    test_text = "title description tags author summary"

    exp = explainer.explain_instance(test_text, dense_predict, num_features=5, num_samples=50)

    return exp


def dense_lime(model_type, model_path, output_file, mode, rel_mode, pooling_path, annotation_file, dataset_info_path, queries_path):
    with open(pooling_path, 'r') as f:
        retrieved_results = json.load(f)

    lime_qrel_results = []
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    tol_num = len(data)
    tol_num = 200
    for _id in tqdm(range(1, tol_num + 1)):
        rel_info = get_rel_info_by_id(annotation_file, _id - 1)
        query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id - 1)
        pair_id = get_pair_id_by_id(annotation_file, _id - 1)
        ind = 500
        if rel_mode == 'qexp':
            ind = 0
        elif rel_mode == 'dexp':
            ind = 2
        if rel_info[ind] != '0':
            lime_result = dense_lime_explainer(model_type, model_path, retrieved_results[str(pair_id)], annotation_file, dataset_info_path, queries_path, _id - 1, mode)
            if lime_result is None:
                continue
            # lime_result.save_to_file(os.path.join(tmp_dir, f'{_id}.html'))
            lime_result = lime_result.as_list()
            explain = {}
            for feature in lime_result:
                explain[feature[0]] = feature[1]
            lime_qrel_results.append({'id': _id, "query": query, "input_id": input_id,
                                      'candidate_id': candidate_id, 'explanation': explain})

    with open(output_file, 'w') as f:
        json.dump(lime_qrel_results, f, indent=4)


if __name__ == '__main__':
    model_type = 'ColBERTv2'
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
    outfile = os.path.join(output_dir, f'ColBERTv2_lime_exp_{mode}_{rel}.json')
    pooling = args.pooling_path
    annotation = args.annotation
    dataset_info = args.dataset_info
    queries_pth = args.queries
    print(mode, rel)
    dense_lime(model_type, model_path, outfile, mode, rel, pooling, annotation, dataset_info, queries_pth)


