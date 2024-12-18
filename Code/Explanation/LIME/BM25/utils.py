import json
import numpy as np


def get_query_dataset_by_id(file_path, queries, _id):
    with open(file_path, 'r') as f:
        data = json.load(f)
    qid_to_text = {}
    with open(queries, 'r') as f:
        for line in f:
            line = line.split('\t')
            qid_to_text[line[0]] = line[1].strip()
    r = data[_id]
    query = qid_to_text[r['query_id']]
    dataset_id1 = r['target_dataset_id']
    dataset_id2 = r['candidate_dataset_id']
    # cursor = get_cursor(db)
    # sql = f'select query, dataset_id1, dataset_id2 from {table_name} WHERE `id` = {_id}'
    # exec_res = execute_sql(sql, cursor)
    return [query, dataset_id1, dataset_id2]


def get_rel_info_by_id(file_path, _id):
    with open(file_path, 'r') as f:
        data = json.load(f)
    r = data[_id]
    rel_info = [str(r['qrel']), r['query_explanation'], str(r['drel']), r['dataset_explanation']]
    return rel_info

def get_dataset_info_by_id(dataset_info, _id):
    with open(dataset_info, 'r') as f:
        data = json.load(f)
    for d in data:
        if d['id'] == _id:
            r = d
            break

    dataset_info = {'title': r['title'],
                    'description': r['description'],
                    'tags': ','.join(r['tags']),
                    'author': r['author'],
                    'summary': r['summary']}
    return dataset_info


def get_pair_id_by_id(file_path, _id):
    with open(file_path, 'r') as f:
        data = json.load(f)
    r = data[_id]
    pair_id = int(r['qdpair_id'])

    return pair_id


def dataset_info_from_fields(fields_str, dataset):
    fields = [f for f in fields_str.split(' ') if len(f) > 0]
    dataset_info = []
    for field in fields:
        dataset_info.append(dataset[field])
    return "\n".join(dataset_info)


def get_pair_info(dataset_info, q, d1, d2, mode):
    input_dataset_info = get_dataset_info_by_id(dataset_info, d1)
    candidate_dataset_info = get_dataset_info_by_id(dataset_info, d2)
    if mode == 'qd':
        pair_info = q + "\n" + "\n".join(input_dataset_info.values())
    elif mode == 'q':
        pair_info = q
    else:
        pair_info = "\n".join(input_dataset_info.values())
    return pair_info, input_dataset_info, candidate_dataset_info


def tokenized_query(q):
    return q.split()


def get_pooling_result(file_path, q, d1):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    for r in json_data:
        if r['query_dataset_id'] == d1 and r['query'] == q:
            return list(r['candidates'][0].keys())


def get_pooling_max_score(file_path, q, d1):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    for r in json_data:
        if r['query_dataset_id'] == d1 and r['query'] == q:
            return list(r['candidates'][0].values())[0]

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    similarity = dot_product / (norm_a * norm_b)
    return similarity


