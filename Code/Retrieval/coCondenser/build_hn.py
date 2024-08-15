from argparse import ArgumentParser
import os
import random
import json


def load_ranking(rank_file, relevance, n_sample, depth):
    with open(rank_file) as rf:
        lines = iter(rf)
        q_0, p_0, _ = next(lines).strip().split()

        curr_q = q_0
        negatives = [] if p_0 in relevance[q_0] else [p_0]

        while True:
            try:
                q, p, _ = next(lines).strip().split()
                if q != curr_q:
                    negatives = negatives[:depth]
                    random.shuffle(negatives)
                    yield curr_q, relevance[curr_q], negatives[:n_sample]
                    curr_q = q
                    negatives = [] if p in relevance[q] else [p]
                else:
                    if p not in relevance[q]:
                        negatives.append(p)
            except StopIteration:
                negatives = negatives[:depth]
                random.shuffle(negatives)
                yield curr_q, relevance[curr_q], negatives[:n_sample]
                return


random.seed(42)

parser = ArgumentParser()
parser.add_argument('--hn_file', required=True)
parser.add_argument('--qrels', required=True)
parser.add_argument('--save_to', required=True)

parser.add_argument('--truncate', type=int, default=128)
parser.add_argument('--n_sample', type=int, default=30)
parser.add_argument('--depth', type=int, default=200)
parser.add_argument('--shard_size', type=int, default=45000)

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

    train_qrels.update(valid_qrels)
    train_qrels = {k: v for k, v in train_qrels.items() if sum(v.values()) > 0}

    counter = 0
    shard_id = 0
    f = None
    save_to_train = os.path.join(args.save_to, 'train-hn')
    os.makedirs(save_to_train, exist_ok=True)

    for query, relevant_docs, negative_samples in load_ranking(args.hn_file, train_qrels, args.n_sample, args.depth):
        train_example = {
            'query': query_info_dict_encoded[query],
            'positives': [dataset_info_dict_encoded[p] for p in relevant_docs],
            'negatives': [dataset_info_dict_encoded[n] for n in negative_samples][:args.n_sample],
        }
        counter += 1
        if f is None:
            f = open(os.path.join(save_to_train, f'split{shard_id:02d}.hn.json'), 'w')
        f.write(json.dumps(train_example) + '\n')
        if counter == args.shard_size:
            f.close()
            f = None
            shard_id += 1
            counter = 0
        
    if f is not None:
        f.close()


if __name__ == "__main__":
    run()