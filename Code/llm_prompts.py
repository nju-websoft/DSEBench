llm_annotator_prompt = """
"I want you to act as an explainable dataset recommender. You task is to make judgments (0: not recommended or neutral, 1: recommended, 2: highly recommended) on whether to recommend a candidate dataset to a user based on user's query and a dataset that user already used, respectively, and choose some features of the candidate dataset to explain why you give the judgements. If your score is 1 or 2, explain that score by some most important candidate dataset keys (choose from: title, description, tags, author and summary), but not output specific values. If your score is 0, the output key is None. Clear scores must be given. Here's the candidate dataset: {candidate_dataset}. User's query: {user_query}. First, you need to give a judgment(0, 1 or 2), namely judgment_1, to the candidate dataset, based on whether the user's query and the candidate dataset have same theme ONLY and strictly, and explain why. Dataset that user already used: {input_dataset}. Then you need to give another judgment (0, 1 or 2), namely judgment_2, to the candidate dataset based on Used Dataset ONLY and strictly, and explain why. Your output should be strictly formatted as :[judgment_1]-[keys_1];[judgment_2]-[keys_2]."
"""

llm_reranking_zero_shot_prompt = """
You are given a keyword query and a target dataset. Based on these inputs, a search system has already retrieved a set of candidate datasets. Your task is to re-rank these candidate datasets so that those most relevant to the input are listed first. Each dataset has a unique ID and some descriptive fields (Title, Description, Tags, Author, Summary).

Please rank all the candidate datasets from most to least relevant to the keyword query and the target dataset. The output should be a list of IDs in the format: [ID_1, ID_2, …, ID_{len(dataset_ids)}] without any additional words or explanations.

Here are the inputs:
{query_info}

And here are the candidate datasets, which are separated by `{separator}`:
{dataset_info}

Now please provide the ranked list of IDs in the specified format: `[ID_1, ID_2, …, ID_{len(dataset_ids)}]`. Don't output other words.
"""

llm_reranking_one_shot_prompt = """
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

llm_explanation_prompt = """
Please generate dataset-dataset relevance explanation between input dataset and candidate dataset by some candidate dataset fields (choose from: title, description, tags, author and summary). Here is the input dataset: {input_info}. Here is the candidate dataset: {candidate_dataset_info}. Your output should be strictly formatted as: [keys_1, keys_2, ...]. Square brackets cannot be missing. Don't output other words.
"""