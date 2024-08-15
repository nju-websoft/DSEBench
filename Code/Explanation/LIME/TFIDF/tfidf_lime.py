import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from lime.lime_text import LimeTextExplainer
import numpy as np
from utils import get_query_dataset_by_id, dataset_info_from_fields, get_rel_info_by_id
from utils import get_pair_info, cosine_similarity, get_pair_id_by_id
import argparse
import os
import json


def get_top_scores(annotation_file, queries_path, dataset_info_path, all_tfidf, model):
    top_scores = {}
    with open(annotation_file, "r") as f:
        data = json.load(f)
    for pair_id in tqdm(range(1, 142)):
        ids = []
        for i, d in enumerate(data):
            if d['qdpair_id'] == str(pair_id):
                ids.append(i)
        _id = ids[0]
        query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id)
        pair_info, input_dataset_info, candidate_dataset_info = get_pair_info(dataset_info_path, query, input_id, candidate_id, mode)
        q_embed = model.transform([pair_info]).toarray()[0]
        cosine_sim = []
        for ind, item in enumerate(all_tfidf):
            cosine_sim.append((ind, cosine_similarity(q_embed, item)))
        re2 = sorted(cosine_sim, key=lambda x: x[1], reverse=True)[:20]
        top_score = re2[0][1]
        for _id in ids:
            top_scores[_id] = top_score
        print(top_scores)
    return top_scores


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



def tfidf_lime_explainer(_id, model, top_scores, pooling_path, annotation_file, dataset_info_path, queries_path):
    query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id)

    pair_id = get_pair_id_by_id(annotation_file, _id)
    with open(pooling_path, 'r') as f:
        data = json.load(f)
    tfidf_pooling = data[str(pair_id)]
    if candidate_id not in tfidf_pooling:
        return None
    pair_info, input_dataset_info, candidate_dataset_info = get_pair_info(dataset_info_path, query, input_id, candidate_id, mode)
    q_vector = model.transform([pair_info]).toarray()[0]
    top_score = top_scores[_id]
    def tfidf_predict(texts): #input: a list of string   output: a list of probability [[0.6,0.4] [0.7,0.2]]
        labels = []

        for text in texts:
            print(text)
            if len(text.strip()) == 0:
                labels.append([0.0, 1.0])
                continue
            new_dataset_info = dataset_info_from_fields(text, candidate_dataset_info)
            print(new_dataset_info)

            d_vector = model.transform([new_dataset_info]).toarray()[0]
            score = cosine_similarity(q_vector, d_vector)
            p_rel = min(1 - (top_score - score) / top_score, 1)
            labels.append([p_rel, 1-p_rel])

        return np.array(labels)

    explainer = LimeTextExplainer(class_names=['relevant', 'irrelevant'])

    test_text = "title description tags author summary"

    exp = explainer.explain_instance(test_text, tfidf_predict, num_features=5, num_samples=50)
    return exp


def tfidf_lime(output_file, rel_mode, pooling_path, annotation_file, dataset_info_path, queries_path):

    ori_corpus, ori_tokenized_corpus, doc_ids = get_corpus(dataset_info_path)
    lime_qrel_results = []
    vectorizer = TfidfVectorizer()
    vectorizer.fit(ori_corpus)
    tfidf_embedding = vectorizer.transform(ori_corpus).toarray()
    top_scores = get_top_scores(annotation_file, queries_path, dataset_info_path, tfidf_embedding, vectorizer)
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    tol_num = len(data)
    # print(tol_num)
    for _id in tqdm(range(1, tol_num+1)):
        rel_info = get_rel_info_by_id(annotation_file, _id-1)
        query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id-1)
        ind = 500
        if rel_mode == 'qexp':
            ind = 0
        elif rel_mode == 'dexp':
            ind = 2
        if rel_info[ind] != '0':
            lime_result = tfidf_lime_explainer(_id-1, vectorizer, top_scores, pooling_path, annotation_file, dataset_info_path, queries_path)
            if lime_result is None:
                continue
            lime_result = lime_result.as_list()
            explain = {}
            for feature in lime_result:
                explain[feature[0]] = feature[1]
            lime_qrel_results.append({'id': _id, "query": query, "input_id": input_id,
                                      'candidate_id': candidate_id, 'explanation': explain})

    with open(output_file, 'w') as f:
        json.dump(lime_qrel_results, f, indent=4)


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
    outfile = os.path.join(output_dir, f'tfidf_lime_exp_{mode}_{rel}.json')
    pooling = args.pooling_path
    annotation = args.annotation
    dataset_info = args.dataset_info
    queries = args.queries
    tfidf_lime(outfile, rel, pooling, annotation, dataset_info, queries)


