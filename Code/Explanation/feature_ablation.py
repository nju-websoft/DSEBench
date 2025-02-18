import re
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from ragatouille import RAGPretrainedModel
from transformers import AutoTokenizer
import subprocess
from tqdm import tqdm


fields = ['title', 'description', 'tags', 'author', 'summary']


def read_dataset_metadata(filename='dataset_metadata.json', sep='\n'):
    dataset_info_dict = {}
    with open(filename, 'r') as f:
        data = json.load(f)
    for item in data:
        dataset_id = item[0]
        info = sep.join(item[1:])
        dataset_info_dict[dataset_id] = info
    return dataset_info_dict


def replace_sep(text, sep='\n'):
    pattern = r' \[SEP\] '
    return sep.join(re.split(pattern, text))


def read_query_info(filename, dataset_info_dict):
    query_info_dict = {}
    with open(filename, 'r') as f:
        data = json.load(f)
        for item in data:
            query_info_dict[item['id']] = {
                'content': item['content'],
                'query': item['keywords'],
                'dataset': dataset_info_dict[item['dataset_id']],
            }
    return query_info_dict


def read_dataset_mask_info(filename='dataset_mask_info.json', remove_sep=True):
    dataset_mask_info_dict = {}
    pattern = r' \[SEP\] | '
    with open(filename, 'r') as f:
        data = json.load(f)
        if not remove_sep:
            dataset_mask_info_dict = data
        else:
            for dataset_id, mask_dict in data.items():
                dataset_mask_info_dict[dataset_id] = {k: replace_sep(v) for k, v in mask_dict.items()}
    return dataset_mask_info_dict


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


def read_explain_pair(run_results_filename, qrels_filename):
    with open(run_results_filename, 'r') as f:
        run_dict = json.load(f)
    
    qrels_kw, qrels_ds = read_qrels_origin(qrels_filename)

    exp_pair= []

    for query_id, rel_dict in run_dict.items():
        for dataset_id, score in rel_dict.items():
            if qrels_kw[query_id].get(dataset_id, 0) > 0 or qrels_ds[query_id].get(dataset_id, 0) > 0:
                exp_pair.append((query_id, dataset_id, score))
    return exp_pair


def similarity_bm25(bm25, document_a, document_b):
    """Computes BM25 score of given `document A` in relation to given `document B` .

    Parameters
    ----------
    bm25: BM25Okapi
    document_a : list of str
        Document to be scored.
    document_b : list of str
        Document to be scored.
    Returns
    -------
    float
        BM25 score.

    """
    PARAM_K1 = bm25.k1
    PARAM_B = bm25.b
    EPSILON = bm25.epsilon
    
    score = 0
    doc_freqs = {}
    for word in document_b:
        if word not in doc_freqs:
            doc_freqs[word] = 0
        doc_freqs[word] += 1
    freq = 1
    default_idf = math.log(bm25.corpus_size - freq + 0.5) - math.log(freq + 0.5)
    for word in document_a:
        if word not in doc_freqs:
            continue
        score += (bm25.idf.get(word,default_idf) * doc_freqs[word] * (PARAM_K1 + 1)
                  / (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * len(document_b) / bm25.avgdl)))
    return score


def bm25_explain(save_path):
    dataset_info_dict = read_dataset_metadata(sep='\n')
    query_info_dict_all = read_query_info('shap_lime/query_info.json', dataset_info_dict)
    dataset_mask_info_dict = read_dataset_mask_info(remove_sep=True)

    run_results_filename = 'shap_lime/results/bm25_results.json'
    qrels_filename = '../dense/coCondenser/qrels.txt'
    exp_pair = read_explain_pair(run_results_filename, qrels_filename)
    
    dataset_ids, corpus = dataset_info_dict.keys(), dataset_info_dict.values()
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    def compute_mask_scores(query_info_dict):
        results_pair = {}
        for query_id, dataset_id, score in exp_pair:
            tokenized_query = replace_sep(query_info_dict[query_id], sep='\n').split()
            mask_info_dict = dataset_mask_info_dict[dataset_id]
            tokenized_dataset = replace_sep(mask_info_dict['full'], '\n').split()
            similarity_full = similarity_bm25(bm25, tokenized_query, tokenized_dataset) + np.finfo(np.float64).eps
            if query_id not in results_pair.keys():
                results_pair[query_id] = {}
            results_pair[query_id][dataset_id] = {'full': similarity_full}
            for field in fields:
                mask_info = mask_info_dict[field]
                tokenized_doc = replace_sep(mask_info, '\n').split()
                similarity_mask = similarity_bm25(bm25, tokenized_query, tokenized_doc)
                results_pair[query_id][dataset_id][field] = similarity_mask / similarity_full
        return results_pair
    
    query_info_dict_qd = {k: v['content'] for k, v in query_info_dict_all.items()}
    query_info_dict_q = {k: v['query'] for k, v in query_info_dict_all.items()}
    query_info_dict_d = {k: v['dataset'] for k, v in query_info_dict_all.items()}
    results = {
        'query-dataset': compute_mask_scores(query_info_dict_qd),
        'query': compute_mask_scores(query_info_dict_q),
        'dataset': compute_mask_scores(query_info_dict_d),
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def tfidf_explain(save_path):
    dataset_info_dict = read_dataset_metadata(sep='\n')
    query_info_dict_all = read_query_info('shap_lime/query_info.json', dataset_info_dict)
    dataset_mask_info_dict = read_dataset_mask_info(remove_sep=True)

    run_results_filename = 'shap_lime/results/tfidf_results.json'
    qrels_filename = '../dense/coCondenser/qrels.txt'
    exp_pair = read_explain_pair(run_results_filename, qrels_filename)
    
    dataset_ids, corpus = dataset_info_dict.keys(), dataset_info_dict.values()

    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)

    def compute_mask_scores(query_info_dict):
        results_pair = {}
        for query_id, dataset_id, score in exp_pair:
            query_info = replace_sep(query_info_dict[query_id], sep='\n')
            mask_info_dict = dataset_mask_info_dict[dataset_id]
            dataset_info = replace_sep(mask_info_dict['full'], '\n')
            query_vector = vectorizer.transform([query_info])
            dataset_vector = vectorizer.transform([dataset_info])
            similarity_full = cosine_similarity(query_vector, dataset_vector).tolist()[0][0] + np.finfo(np.float64).eps
            if query_id not in results_pair.keys():
                results_pair[query_id] = {}
            results_pair[query_id][dataset_id] = {'full': similarity_full}
            for field in fields:
                mask_info = replace_sep(mask_info_dict[field], '\n')
                mask_vector = vectorizer.transform([mask_info])
                similarity_mask = cosine_similarity(query_vector, mask_vector).tolist()[0][0]
                results_pair[query_id][dataset_id][field] = similarity_mask / similarity_full
        return results_pair

    query_info_dict_qd = {k: v['content'] for k, v in query_info_dict_all.items()}
    query_info_dict_q = {k: v['query'] for k, v in query_info_dict_all.items()}
    query_info_dict_d = {k: v['dataset'] for k, v in query_info_dict_all.items()}
    results = {
        'query-dataset': compute_mask_scores(query_info_dict_qd),
        'query': compute_mask_scores(query_info_dict_q),
        'dataset': compute_mask_scores(query_info_dict_d),
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def pretrained_model_explain(model_name, run_results_filename, save_path):
    dataset_info_dict = read_dataset_metadata(sep=' [SEP] ')
    query_info_dict_all = read_query_info('shap_lime/query_info.json', dataset_info_dict)
    dataset_mask_info_dict = read_dataset_mask_info(remove_sep=False)

    qrels_filename = '../dense/coCondenser/qrels.txt'
    exp_pair = read_explain_pair(run_results_filename, qrels_filename)

    if 'stella_en_1.5B_v5' in model_name:
        model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
        def encode_query(texts: list):
            embeddings = model.encode(texts, normalize_embeddings=True, prompt_name="s2p_query")
            return embeddings
    elif 'thenlper/gte-large' in model_name:
        model = SentenceTransformer(model_name)
        def encode_query(texts: list):
            embeddings = model.encode(texts, normalize_embeddings=True)
            return embeddings
    elif 'bge-large-en-v1.5' in model_name:
        model = SentenceTransformer(model_name)
        def encode_query(texts: list):
            instruction = "Represent this sentence for searching relevant passages:"
            texts = [instruction+q for q in texts]
            embeddings = model.encode(texts, normalize_embeddings=True)
            return embeddings

    def compute_mask_scores(query_info_dict):
        results_pair = {}
        for query_id, dataset_id, score in exp_pair:
            query_info = query_info_dict[query_id]
            mask_info_dict = dataset_mask_info_dict[dataset_id]
            dataset_info = mask_info_dict['full']
            query_vector = encode_query([query_info])
            dataset_vector = model.encode([dataset_info], normalize_embeddings=True)
            similarity = query_vector @ dataset_vector.T
            similarity_full = similarity[0, 0] + np.finfo(np.float64).eps
            if query_id not in results_pair.keys():
                results_pair[query_id] = {}
            results_pair[query_id][dataset_id] = {'full': similarity_full}
            for field in fields:
                mask_info = mask_info_dict[field]
                mask_vector = model.encode([mask_info], normalize_embeddings=True)
                similarity = query_vector @ mask_vector.T
                similarity_mask = similarity[0, 0]
                results_pair[query_id][dataset_id][field] = similarity_mask / similarity_full
        return results_pair

    query_info_dict_qd = {k: v['content'] for k, v in query_info_dict_all.items()}
    query_info_dict_q = {k: v['query'] for k, v in query_info_dict_all.items()}
    query_info_dict_d = {k: v['dataset'] for k, v in query_info_dict_all.items()}
    results = {
        'query-dataset': compute_mask_scores(query_info_dict_qd),
        'query': compute_mask_scores(query_info_dict_q),
        'dataset': compute_mask_scores(query_info_dict_d),
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def tevatron_explain(ckpt_path, run_results_filename, save_path, temp_data_path):
    dataset_info_dict = read_dataset_metadata(sep=' [SEP] ')
    query_info_dict_all = read_query_info('shap_lime/query_info.json', dataset_info_dict)
    dataset_mask_info_dict = read_dataset_mask_info(remove_sep=False)

    qrels_filename = '../dense/coCondenser/qrels.txt'
    exp_pair = read_explain_pair(run_results_filename, qrels_filename)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)

    def compute_mask_scores():
        query_info_dict_all_encoded = {outer_key: {inner_key: tokenizer.encode(inner_value, add_special_tokens=False, max_length=512, truncation=True) for inner_key, inner_value in inner_dict.items()} for outer_key, inner_dict in query_info_dict_all.items()}

        results_pair = {}
        with tqdm(total=len(exp_pair)) as pbar:
            for query_id, dataset_id, score in exp_pair:
                mask_info_dict = dataset_mask_info_dict[dataset_id]
                mask_info_dict_encoded = {k: tokenizer.encode(v, add_special_tokens=False, max_length=512, truncation=True) for k, v in mask_info_dict.items()}
                # build corpus
                corpus_path = f'{temp_data_path}/bert/corpus'
                os.makedirs(corpus_path, exist_ok=True)
                f = open(os.path.join(corpus_path, 'split00.json'), 'w')
                for document_id, document_encoded in mask_info_dict_encoded.items():
                    encoded = {
                        'text_id': document_id,
                        'text': document_encoded
                    }
                    f.write(json.dumps(encoded) + '\n')
                f.close()
                # build query
                query_path = f'{temp_data_path}/bert/query'
                os.makedirs(query_path, exist_ok=True)
                f = open(os.path.join(query_path, 'query.json'), 'w')
                query_info_dict = query_info_dict_all_encoded[query_id]
                for key, text in query_info_dict.items():
                    encoded = {
                        'text_id': f'{query_id}_{key}',
                        'text': text
                    }
                    f.write(json.dumps(encoded) + '\n')
                f.close()
                # encode corpus
                encoding_path = f'{temp_data_path}/encoding'
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
                tsv_path = f'{temp_data_path}/rank.tsv'
                command = [
                    "python", "-m", "tevatron.faiss_retriever",
                    "--query_reps", os.path.join(encoding_path, 'query', 'qry.pt'),
                    "--passage_reps", os.path.join(encoding_path, 'corpus', 'split00.json'),
                    "--depth", str(len(mask_info_dict)),
                    "--batch_size", "-1",
                    "--save_text", 
                    "--save_ranking_to", tsv_path,
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                search_scores = {}
                with open(tsv_path, 'r') as f:
                    for line in f:
                        if line:
                            query_id, document_id, score = line.strip().split()
                            if query_id not in search_scores:
                                search_scores[query_id] = {}
                            search_scores[query_id][document_id] = float(score)
                for query_id in search_scores.keys():
                    similarity_full = search_scores[query_id]['full'] + np.finfo(np.float64).eps
                    if query_id not in results_pair.keys():
                        results_pair[query_id] = {}
                    results_pair[query_id][dataset_id] = {'full': similarity_full}
                    for field in fields:
                        results_pair[query_id][dataset_id][field] = search_scores[query_id][field] / similarity_full
                pbar.update(1)
        return results_pair

    results_pair = compute_mask_scores()

    query_info_dict_qd, query_info_dict_q, query_info_dict_d = {}, {}, {}
    for k, v in results_pair.items():
        query_id, query_type = k.split('_')
        if query_type == 'content':
            query_info_dict_qd[query_id] = v
        elif query_type == 'query':
            query_info_dict_q[query_id] = v
        elif query_type == 'dataset':
            query_info_dict_d[query_id] = v

    results = {
        'query-dataset': query_info_dict_qd,
        'query': query_info_dict_q,
        'dataset': query_info_dict_d,
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


def ColBERTv2_explain(ckpt_path, run_results_filename, save_path):
    RAG = RAGPretrainedModel.from_pretrained(ckpt_path)
    dataset_info_dict = read_dataset_metadata(sep=' [SEP] ')
    query_info_dict_all = read_query_info('shap_lime/query_info.json', dataset_info_dict)
    dataset_mask_info_dict = read_dataset_mask_info(remove_sep=False)

    qrels_filename = '../dense/coCondenser/qrels.txt'
    exp_pair = read_explain_pair(run_results_filename, qrels_filename)

    def compute_mask_scores(query_info_dict):
        results_pair = {}
        for query_id, dataset_id, score in exp_pair:
            query_info = query_info_dict[query_id]
            mask_info_dict = dataset_mask_info_dict[dataset_id]
            search_results = RAG.rerank(
                query=query_info, 
                documents=list(mask_info_dict.values()), 
                k=len(mask_info_dict)
            )
            if query_id not in results_pair.keys():
                results_pair[query_id] = {}
            results_pair[query_id][dataset_id] = {}
            mask_info_dict_keys = list(mask_info_dict.keys())
            for sr in search_results:
                field = mask_info_dict_keys[sr['result_index']]
                score = sr['score']
                results_pair[query_id][dataset_id][field] = score
            score_full = results_pair[query_id][dataset_id]['full']
            results_pair[query_id][dataset_id] = {k: v/score_full for k, v in results_pair[query_id][dataset_id].items() if k in fields}
        return results_pair

    query_info_dict_qd = {k: v['content'] for k, v in query_info_dict_all.items()}
    query_info_dict_q = {k: v['query'] for k, v in query_info_dict_all.items()}
    query_info_dict_d = {k: v['dataset'] for k, v in query_info_dict_all.items()}
    results = {
        'query-dataset': compute_mask_scores(query_info_dict_qd),
        'query': compute_mask_scores(query_info_dict_q),
        'dataset': compute_mask_scores(query_info_dict_d),
    }

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    bm25_explain(save_path='output/bm25_explain_ablation.json')
    # tfidf_explain(save_path='output/tfidf_explain_ablation.json')
    # pretrained_model_explain(
    #     model_name='BAAI/bge-large-en-v1.5',
    #     run_results_filename='shap_lime/results/bge_results.json',
    #     save_path='output/bge_explain_ablation.json'
    # )
    # pretrained_model_explain(
    #     model_name='thenlper/gte-large',
    #     run_results_filename='shap_lime/results/gte_results.json',
    #     save_path='output/gte_explain_ablation.json'
    # )

    # ckpt_path = '/home/qshi/ds_rec_benchmark/datasetRec/dense/coCondenser/coCondenser_ckpt'
    # run_results_path = '/home/qshi/ds_rec_benchmark/datasetRec/dense/coCondenser/results/coCondenser'
    # for fold in list(range(5)) + [100]:  # 
    #     print(f'coCondenser fold: {fold}')
    #     tevatron_explain(
    #         ckpt_path=os.path.join(ckpt_path, f'fold_{fold}'),
    #         run_results_filename=os.path.join(run_results_path, f'fold_{fold}.json'),
    #         save_path=f'output/coCondenser_explain_ablation_fold_{fold}.json',
    #         temp_data_path='.temp_ccds',
    #     )
    
    # ckpt_path = '/home/qshi/ds_rec_benchmark/datasetRec/dense/coCondenser/dpr_ckpt'
    # run_results_path = '/home/qshi/ds_rec_benchmark/datasetRec/dense/coCondenser/results/dpr'
    # for fold in list(range(5)) + [100]:  # 
    #     print(f'dpr fold: {fold}')
    #     tevatron_explain(
    #         ckpt_path=os.path.join(ckpt_path, f'fold_{fold}'),
    #         run_results_filename=os.path.join(run_results_path, f'fold_{fold}.json'),
    #         save_path=f'output/dpr_explain_ablation_fold_{fold}.json',
    #         temp_data_path='.temp_dpr',
    #     )

    # for fold in list(range(5)) + [100]:  # 
    #     print(f'fold: {fold}')
    #     ColBERTv2_explain(
    #         ckpt_path=f'checkpoints/fold_{fold}',
    #         run_results_filename=f'results/ColBERTv2/fold_{fold}.json',
    #         save_path=f'explain/colbert_explain_ablation_fold_{fold}.json'
    #     )


    # # without ft
    # ckpt_path = 'Luyu/co-condenser-marco-retriever'
    # run_results_filename = '../dense/coCondenser/results/fold_100_coCondenser_without_ft.json'
    # tevatron_explain(
    #     ckpt_path=ckpt_path,
    #     run_results_filename=run_results_filename,
    #     save_path=f'output/coCondenser_explain_ablation_fold_200.json',
    #     temp_data_path='.temp_ccds',
    # )

    # ckpt_path = '../dense/tevatron/tevatron/examples/dpr/model_nq'
    # run_results_filename = '../dense/coCondenser/results/fold_100_dpr_without_ft.json'
    # tevatron_explain(
    #     ckpt_path=ckpt_path,
    #     run_results_filename=run_results_filename,
    #     save_path=f'output/dpr_explain_ablation_fold_200.json',
    #     temp_data_path='.temp_dpr',
    # )

    # ckpt_path = 'colbert-ir/colbertv2.0'
    # run_results_filename = '../dense/coCondenser/results/fold_100_ColBERTv2_without_ft.json'
    # ColBERTv2_explain(
    #     ckpt_path=ckpt_path,
    #     run_results_filename=run_results_filename,
    #     save_path=f'output/colbert_explain_ablation_fold_200.json',
    # )
    
    pass