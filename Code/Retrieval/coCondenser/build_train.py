from argparse import ArgumentParser
import os
import random
import json


random.seed(42)

parser = ArgumentParser()
parser.add_argument('--qrels', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--hn_file', type=str, required=True)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--shard_size', type=int, default=45000)
parser.add_argument('--merge_valid', type=int, default=0)

args = parser.parse_args()


def run():
    with open('query_info_encoded.json', 'r') as f:
        query_info_dict_encoded = json.load(f)

    with open('dataset_info_encoded.json', 'r') as f:
        dataset_info_dict_encoded = json.load(f)
    
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

    # build train
    counter = 0
    shard_id = 0
    f = None
    save_to_train = os.path.join(args.save_to, 'train')
    os.makedirs(save_to_train, exist_ok=True)

    hn_dict = {}
    with open(args.hn_file, 'r') as hn_f:
        for line in hn_f:
            if line:
                query_id, dataset_id, _ = line.strip().split('\t')
                if query_id not in hn_dict.keys():
                    hn_dict[query_id] = []
                if dataset_id not in train_qrels[query_id].keys():
                    hn_dict[query_id].append(dataset_id)

    for query_id, rel_dict in train_qrels.items():
        pp, nn = [], []
        for dataset_id, rel in rel_dict.items():
            if rel == 0:
                nn.append(dataset_info_dict_encoded[dataset_id])
            else:
                pp.append(dataset_info_dict_encoded[dataset_id])
        # supplement negative data
        if len(nn) == 0:
            nn = [dataset_info_dict_encoded[x] for x in hn_dict[query_id]]
        random.shuffle(nn)
        train_example = {
            'query': query_info_dict_encoded[query_id],
            'positives': pp,
            'negatives': nn[:args.n_sample],
        }
        counter += 1
        if f is None:
            f = open(os.path.join(save_to_train, f'split{shard_id:02d}.json'), 'w')
        f.write(json.dumps(train_example) + '\n')
        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0
    
    if f is not None:
        f.close()

    # # build query
    # save_to_query = os.path.join(args.save_to, 'query')
    # os.makedirs(save_to_query, exist_ok=True)

    # f = open(os.path.join(save_to_query, f'train.query.json'), 'w')
    # for query_id in train_qrels.keys():
    #     encoded = {
    #         'text_id': query_id,
    #         'text': query_info_dict_encoded[query_id]
    #     }
    #     f.write(json.dumps(encoded) + '\n')
    # f.close()

    # f = open(os.path.join(save_to_query, f'dev.query.json'), 'w')
    # for query_id in valid_qrels.keys():
    #     encoded = {
    #         'text_id': query_id,
    #         'text': query_info_dict_encoded[query_id]
    #     }
    #     f.write(json.dumps(encoded) + '\n')
    # f.close()

    # f = open(os.path.join(save_to_query, f'test.query.json'), 'w')
    # for query_id in test_qrels.keys():
    #     encoded = {
    #         'text_id': query_id,
    #         'text': query_info_dict_encoded[query_id]
    #     }
    #     f.write(json.dumps(encoded) + '\n')
    # f.close()


if __name__ == "__main__":
    run()

