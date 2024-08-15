from argparse import ArgumentParser
import os
import random
import json


random.seed(42)

parser = ArgumentParser()
parser.add_argument('--qrels', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--merge_valid', type=int, default=0)

args = parser.parse_args()


def run():
    with open('query_info_encoded.json', 'r') as f:
        query_info_dict_encoded = json.load(f)
    
    with open(os.path.join(args.qrels, 'train.json'), 'r') as f:
        train_qrels = json.load(f)
    
    with open(os.path.join(args.qrels, 'valid.json'), 'r') as f:
        valid_qrels = json.load(f)
    
    with open(os.path.join(args.qrels, 'test.json'), 'r') as f:
        test_qrels = json.load(f)

    if args.merge_valid == 1:
        train_qrels.update(valid_qrels)
        valid_qrels = test_qrels
    train_qrels = {k: v for k, v in train_qrels.items() if sum(v.values()) > 0}

    # build query
    save_to_query = os.path.join(args.save_to, 'query')
    os.makedirs(save_to_query, exist_ok=True)

    f = open(os.path.join(save_to_query, f'train.query.json'), 'w')
    for query_id in train_qrels.keys():
        encoded = {
            'text_id': query_id,
            'text': query_info_dict_encoded[query_id]
        }
        f.write(json.dumps(encoded) + '\n')
    f.close()

    f = open(os.path.join(save_to_query, f'dev.query.json'), 'w')
    for query_id in valid_qrels.keys():
        encoded = {
            'text_id': query_id,
            'text': query_info_dict_encoded[query_id]
        }
        f.write(json.dumps(encoded) + '\n')
    f.close()

    f = open(os.path.join(save_to_query, f'test.query.json'), 'w')
    for query_id in test_qrels.keys():
        encoded = {
            'text_id': query_id,
            'text': query_info_dict_encoded[query_id]
        }
        f.write(json.dumps(encoded) + '\n')
    f.close()


if __name__ == "__main__":
    run()

