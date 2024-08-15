import os
import json
from zhipuai import ZhipuAI
import pymysql
import re
import argparse
from tqdm import tqdm
from utils import get_query_dataset_by_id, get_dataset_info_by_id, get_rel_info_by_id, get_pair_info


def zhipu_response(prompt, model_name):
    client = ZhipuAI(api_key="xxxxxxxxxxxxx.xxxxxx")  # APIKey
    features = ['title', 'description', 'tags', 'author', 'summary']
    flag, num = 1, 0
    message = ''
    while flag:
        if num > 2:
            break
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    # {"role": "system", "content": "You are a recommend system."},
                    {"role": "user",
                     "content": prompt,
                     },
                ],
            )
            message = response.choices[0].message.content
            msg_l = message.strip('[]').split()
            flag = 0
            num += 1
            for i in msg_l:
                if i.strip(',.;') not in features:
                    flag = 1
                    break
        except:
            num += 1
            flag = 1
            print('ERROR')
        # print(message)
    return message


def process_response(data):
    data = data.strip('[]').split()
    data_list = []
    for d in data:
        data_list.append(d.strip(',;.'))
    return data_list


def compose_prompt(annotation_file, dataset_info_path, queries_path, _id, mode):
    query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id)
    pair_info, input_dataset_info, candidate_dataset_info = get_pair_info(dataset_info_path, query, input_id, candidate_id, mode)
    input_info = " ".join(list(input_dataset_info.values()))
    prompt = f"Please generate dataset-dataset relevance explanation between input dataset and candidate dataset by some candidate dataset fields (choose from: title, description, tags, author and summary). " + \
             f"Here is the input dataset: {input_info}. " + \
             f"Here is the candidate dataset: {candidate_dataset_info}. " + \
             f"Your output should be strictly formatted as: [keys_1, keys_2, ...]. Square brackets cannot be missing. Don't output other words."

    return prompt


def llm_explain(output_file, model_name, rel_type, annotation_file, dataset_info_path, queries_path):
    results = []
    all_tokens = 0
    for _id in tqdm(range(1, 7416)):
        rel_info = get_rel_info_by_id(annotation_file, _id-1)
        query, input_id, candidate_id = get_query_dataset_by_id(annotation_file, queries_path, _id-1)
        ind = 500
        if rel_type == 'qexp':
            ind = 0
        elif rel_type == 'dexp':
            ind = 2
        if rel_info[ind] != '0':
            prompt = compose_prompt(annotation_file, dataset_info_path, queries_path, _id, mode)
            all_tokens += len(prompt.split())/2*3
            msg = zhipu_response(prompt, model_name)
            data = process_response(msg)
            explain = {}
            features = ['title', 'description', 'tags', 'author', 'summary']
            for i, f in enumerate(features):
                if f in data:
                    explain[f] = 1
                else:
                    explain[f] = 0
            results.append({'id': _id, "query": query, "input_id": input_id,
                            'candidate_id': candidate_id, 'explanation': explain})

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    model = 'glm-4-air'

    parser = argparse.ArgumentParser(description="model_path: model path; mode: 'qd' or 'd' or 'q'; rel: 'qexp' or 'dexp'; output_dir: directory to output file; pooling_path: retrieved file path; "
                                                 "annotation: annotation file path; dataset_info: dataset info file path; queries: queries file path")
    parser.add_argument('--mode', help="'qd' or 'd' or 'q'", required=True)
    parser.add_argument('--rel', help="'qexp' or 'dexp'", required=True)
    parser.add_argument('--output_dir', help="directory to output file", required=True)
    parser.add_argument('--annotation', help="annotation file path", required=True)
    parser.add_argument('--dataset_info', help="dataset info file path", required=True)
    parser.add_argument('--queries', help="queries file path", required=True)
    args = parser.parse_args()
    mode = args.mode
    rel = args.rel
    output_dir = args.output_dir
    outfile = os.path.join(output_dir, f'{model}_exp_{mode}_{rel}.json')
    annotation = args.annotation
    dataset_info = args.dataset_info
    queries = args.queries
    llm_explain(outfile, model, rel, annotation, dataset_info, queries)

