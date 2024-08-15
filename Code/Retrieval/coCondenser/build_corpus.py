from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import random
import json
import numpy as np


random.seed(42)

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--shard_size', type=int, default=45000)

args = parser.parse_args()


def run():
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
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    
    query_info_dict_encoded = {k: tokenizer.encode(v, add_special_tokens=False, max_length=args.truncate, truncation=True) for k, v in query_info_dict.items()}
    dataset_info_dict_encoded = {k: tokenizer.encode(v, add_special_tokens=False, max_length=args.truncate, truncation=True) for k, v in dataset_info_dict.items()}

    with open('query_info_encoded.json', 'w') as f:
        json.dump(query_info_dict_encoded, f)
    
    with open('dataset_info_encoded.json', 'w') as f:
        json.dump(dataset_info_dict_encoded, f)

    # build corpus
    counter = 0
    shard_id = 0
    f = None
    save_to_corpus = os.path.join(args.save_to, 'corpus')
    os.makedirs(save_to_corpus, exist_ok=True)

    for dataset_id, info in dataset_info_dict_encoded.items():
        counter += 1
        if f is None:
            f = open(os.path.join(save_to_corpus, f'split{shard_id:02d}.json'), 'w')
        encoded = {
            'text_id': dataset_id,
            'text': info
        }
        f.write(json.dumps(encoded) + '\n')
        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0
    

    # remove empty qrels
    fold_num_list = list(np.arange(5)) + [100]
    for fold in fold_num_list:
        fold_dir = f'fold_{fold}'
        for filename in ['train.json']:
            with open(os.path.join(fold_dir, filename), 'r') as f:
                qrels = json.load(f)
            print(f'{fold} {filename} {len(qrels)}')
            qrels = {k: v for k, v in qrels.items() if sum(v.values()) > 0}
            print(f'{fold} {filename} {len(qrels)}')
            with open(os.path.join(fold_dir, filename), 'w') as f:
                json.dump(qrels, f)


if __name__ == "__main__":
    run()

