from zhipuai import ZhipuAI
import json
import os
from tqdm import tqdm
import re
import random
import numpy as np

def check_llm_response(text, top_k):
    if text[0] != '[':
        print('start', text[0])
        return False
    if len(text.strip('[').strip(']').split(', ')) < top_k:
        print('len', len(text.strip('[').strip(']').split(', ')), top_k)
        return False
    return True

def llm_rerank_few_shot_layer(llm_model_name, api_key, rerank_dict, query_info_dict, dataset_info_dict):
    separator = '='*10
    split_k = 5
    epoch_num = 20

    exapmle_query = 'percentage of americans with drivers license'
    example_dataset_id = 'a1108e5d-4aab-4218-82e0-28f5f3126364'
    example_dataset_id_list = [
        '6bc15606-a807-4540-8b1c-a6e5cbf69bda',
        '2953e1e5-8e2e-4ad8-911a-ccd900d0b280',
        '2375f9c6-84b3-4233-aaa7-a3dcf647fc44',
        '86871107-08d9-40b9-907b-52748e616f1c',
        '211e1d20-e483-4300-9583-b0c0179edf00',
        '4172ad0f-d1d8-498a-95bc-e4c45671bfbb',
        'fe4c5764-c730-40f7-a9ea-879b4b114894',
    ]
    example_dataset_id_list_shuffle = example_dataset_id_list.copy()
    random.seed(42)
    random.shuffle(example_dataset_id_list_shuffle)
    example_query_info = f"- Keyword Query: {exapmle_query}\n- Target Dataset:\n{dataset_info_dict[example_dataset_id]}"
    exapmle_dataset_info = f'\n{separator}\n'.join([dataset_info_dict[dataset_id] for dataset_id in example_dataset_id_list_shuffle])
    example_rank_info = f"[{', '.join(example_dataset_id_list)}]"
    example_info = f"*Input*:\n{example_query_info}\n\nCandidate datasets (separated by `{separator}`):\n{exapmle_dataset_info}\n\n*Output*: {example_rank_info}"


    client = ZhipuAI(api_key=api_key)
    rerank_results = {x: {} for x in rerank_dict.keys()}
    with tqdm(total=len(rerank_dict)) as pbar:
        for query_id, dataset_ids in rerank_dict.items():
            query_info = query_info_dict[query_id]
            for epoch in range(epoch_num):
                pbar.set_postfix(current_epoch=epoch)
                random.seed(epoch)
                shuffle_dataset_ids = dataset_ids.copy()
                random.shuffle(shuffle_dataset_ids)
                dataset_ids_splits = np.array_split(shuffle_dataset_ids, split_k)
                for dis in dataset_ids_splits:
                    while True:
                        dataset_info = f'\n{separator}\n'.join([dataset_info_dict[dataset_id] for dataset_id in dis])
                        listwise_messages = [
                            {
                                "role": "system", 
                                "content": "You are a relevance ranker specializing in re-ranking candidate datasets based on a keyword query and a target dataset description."
                            },
                            {
                                "role": "user", 
                                "content": f"""
You are given a keyword query and a target dataset. Based on these inputs, a search system has already retrieved a set of candidate datasets. Your task is to re-rank these candidate datasets so that those most relevant to the input are listed first. Each dataset has a unique ID and some descriptive fields (Title, Description, Tags, Author, Summary).

Please rank all the candidate datasets from most to least relevant to the keyword query and the target dataset. The output should be a list of IDs in the format: `[ID_1, ID_2, …, ID_{len(dis)}]` without any additional words or explanations. 

Here is an example:
```
{example_info}
```

Now the inputs are:
{query_info}

The candidate datasets (separated by `{separator}`) are:
{dataset_info}

Now please provide the ranked list of IDs in the specified format: [ID_1, ID_2, …, ID_{len(dis)}]. Don't output other words.
                        """
                            },
                        ]
                        response = client.chat.completions.create(
                            model=llm_model_name,
                            messages=listwise_messages,
                        )
                        response_text = response.choices[0].message.content
                        with open('llm_layer/llm.few.log', 'a') as logfile:
                            logfile.write(json.dumps({
                                'query_id': query_id,
                                'epoch': epoch,
                                'dataset_ids': list(dis),
                                'response': response_text,
                            }))
                            logfile.write('\n')
                        response_text = response_text.strip('`').strip()
                        if check_llm_response(response_text, len(dis)//2 + 1):
                            temp_list = response_text.strip('[').strip(']').split(', ')
                            reranked_list = []
                            for it in temp_list:
                                if it not in reranked_list:
                                    reranked_list.append(it)
                            for dataset_id in reranked_list[:len(dis)//2]:
                                rerank_results[query_id][dataset_id] = rerank_results[query_id].get(dataset_id, 0) + 1
                            break
            with open('llm_layer/temp_results.json', 'w') as tempfile:
                json.dump(rerank_results, tempfile, indent=2)
            pbar.update(1)
    return rerank_results
