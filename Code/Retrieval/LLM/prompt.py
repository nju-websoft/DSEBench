zero_shot_prompt = """
You are given a keyword query and a target dataset. Based on these inputs, a search system has already retrieved a set of candidate datasets. Your task is to re-rank these candidate datasets so that those most relevant to the input are listed first. Each dataset has a unique ID and some descriptive fields (Title, Description, Tags, Author, Summary).

Please rank all the candidate datasets from most to least relevant to the keyword query and the target dataset. The output should be a list of IDs in the format: [ID_1, ID_2, …, ID_{len(dataset_ids)}] without any additional words or explanations.

Here are the inputs:
{query_info}

And here are the candidate datasets, which are separated by `{separator}`:
{dataset_info}

Now please provide the ranked list of IDs in the specified format: `[ID_1, ID_2, …, ID_{len(dataset_ids)}]`. Don't output other words.
"""

one_shot_prompt = """
You are given a keyword query and a target dataset. Based on these inputs, a search system has already retrieved a set of candidate datasets. Your task is to re-rank these candidate datasets so that those most relevant to the input are listed first. Each dataset has a unique ID and some descriptive fields (Title, Description, Tags, Author, Summary).

Please rank all the candidate datasets from most to least relevant to the keyword query and the target dataset. The output should be a list of IDs in the format: `[ID_1, ID_2, …, ID_{len(dataset_ids)}]` without any additional words or explanations. 

Here is an example:
```
{example_info}
```

Now the inputs are:
{query_info}

The candidate datasets (separated by `{separator}`) are:
{dataset_info}

Now please provide the ranked list of IDs in the specified format: `[ID_1, ID_2, …, ID_{len(dataset_ids)}]`. Don't output other words.
"""