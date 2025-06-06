# DSEBench

**DSEBench** is a test collection designed to support the evaluation of **Dataset Search with Examples (DSE)**, a task that generalizes two established paradigms: keyword-based dataset search and similarity-based dataset discovery. Given a textual query $q$ and a set of target datasets $D_t$ known to be relevant, the goal of DSE is to retrieve a ranked list $D_c$ of candidate datasets that are both relevant to $q$ and similar to the datasets in $D_t$.

As an extension, **Explainable DSE** further requires identifying, for each result dataset $d \in D_c$, a subset of metadata or content fields that explain its relevance to $q$ and similarity to $D_t$.

For further details, please refer to the accompanying paper.


## Datasets

We reused the 46,615 datasets collected from NTCIR. The "datasets.json" file (available at [Zenodo](https://zenodo.org/records/13309568) provides the `id`, `title`, `description`, `tags`, `author`, and `summary` of each dataset in JSON format.

```json
{ 
  "id": "0000de36-24e5-42c1-959d-2772a3c747e7", 
  "title": "Montezuma National Wildlife Refuge: January - April, 1943", 
  "description": "This narrative report for Montezuma National Wildlife Refuge outlines Refuge accomplishments from January through April of 1943. ...", 
  "tags": ["annual-narrative", "behavior", "populations"], 
  "author": "Fish and Wildlife Service", 
  "summary": "Almost continuous rains during April brought flood conditions to the Clyde River as well as to the refuge storage pool. Cayuga Lake is at its highest level in about ton years. ..."
}
```

Below is an example of how to load and use the `datasets.json` file:

```python
import json

# Load the dataset file
with open('datasets.json', 'r') as f:
    datasets_data = json.load(f)
    
    # Iterate through each judgment
    for dataset in datasets_data:
        dataset_id = dataset['id']  # Get the dataset ID
        title = dataset['title']  # Get the title
        
        # Other code to process the judgment data...
```

## Queries

The "[./Data/queries.tsv](./Data/queries.tsv)" file provides 3,979 keyword queries. Each row represents a query with two "\t"-separated columns: `query_id` and `query_text`. The queries can be divided into two categories: generated queries, created from the metadata of datasets, and NTCIR queries, imported from the English part of the NTCIR dataset. Queries with IDs starting with "GEN_" are generated queries, while those starting with "NTCIR" are NTCIR queries.

Below is an example of how to load and use the `./Data/queries.tsv` file:
```python
# Load the queries file
with open('Data/queries.tsv', 'r') as f:
    # Iterate through each line
    for line in f:
        query_id, query_text = line.split('\t')  # Get the query ID and the query text

        # Other code to process the data...
```

## Test and Training Cases

In DSEBench, each input consists of a case, which includes a query and a set of target datasets that are known to be relevant to the query. The "[./Data/cases.tsv](./Data/cases.tsv)" file provides 141 test cases and 5,699 training cases. Each row represents a case with three "\t"-separated columns: `case_id`, `query_id`, and `target_dataset_id`.

Test cases are identified by a `case_id` composed of pure numbers. These test cases are adapted from highly relevant query-dataset pairs from the NTCIR dataset. The remaining cases are training cases. Among these, those with a `case_id` starting with `l1_` are adapted from partially relevant query-dataset pairs from NTCIR, while those starting with `gen_` are synthetic training cases, where the queries are generated queries.

Below is an example of how to load and use the `./Data/cases.tsv` file:
```python
# Load the cases file
with open('Data/cases.tsv', 'r') as f:
    # Iterate through each line
    for line in f:
        case_id, query_id, target_dataset_id = line.split('\t')  # Get the case ID, the query ID, and the target dataset ID

        # Other code to process the data...
```


## Relevance Judgments

The "[./Data/human_annotated_judgments.json](./Data/human_annotated_judgments.json)" file contains 7,415 human-annotated judgments, and the "[./Data/llm_annotated_judgments.json](./Data/llm_annotated_judgments.json)" file contains 122,585 judgments generated by a large language model (LLM). Each JSON object has eight keys: `query_id`, `target_dataset_id`, `candidate_dataset_id`, `case_id` (the ID of the input), `query_rel` (relevance of the candidate dataset to the query, 0: irrelevant; 1: partially relevant; 2: highly relevant), `field_query_rel`, `target_sim` (similarity of the candidate dataset to the target datasets, 0: dissimilar; 1: partially similar; 2: highly similar), and `field_target_sim`. The `field_query_rel` and `field_target_sim` are both lists of length 5 consisting of 0 and 1, corresponding to the fields `[title, description, tags, author, summary]`.

```json
{
    "query_id": "NTCIR_200000", 
    "target_dataset_id": "002ece58-9603-43f1-8e2e-54e3d9649e84", 
    "candidate_dataset_id": "99e3b6a2-d097-463f-b6e1-3caceff300c9", 
    "case_id": "1", 
    "query_rel": 1, 
    "field_query_rel": [1, 1, 1, 0, 0], 
    "target_sim": 2, 
    "field_target_sim": [1, 1, 1, 1, 1]
}
```

Below is an example of how to load and use the `./Data/human_annotated_judgments.json` file:

```python
import json

# Load the judgments file
with open('Data/human_annotated_judgments.json', 'r') as f:
    judgments_data = json.load(f)
    
    # Iterate through each judgment
    for judgment in judgments_data:
        case_id = judgment['case_id']  # Get the case ID
        candidate_dataset_id = judgment['candidate_dataset_id']  # Get the candidate dataset ID
        query_rel = judgment['query_rel']  # Get the query relevance score
        field_query_rel = judgment['field_query_rel']  # Get the field-level query relevance scores (title, description, tags, author, summary)
        
        # Other code to process the judgment data...
```

## Splits for Training, Validation, and Test Sets

To ensure that evaluation results are comparable, we provide predefined train-validation-test splits. The "[./Data/Splits/5-Fold_split](./Data/Splits/5-Fold_split)" folder contains five sub-folders, each providing three qrel files for training, validation, and test sets. The "[./Data/Splits/Annotators_split](./Data/Splits/Annotators_split)" folder contains three qrel files for the training, validation, and test sets as well.

These files are used in the same way as the relevance judgments files.

## Baselines

### Baselines for Retrieval and Reranking

For **retrieval**, we evaluated two sparse retrieval models: (1) TF-IDF, (2) BM25 and five dense retrieval models: (3) BGE (bge-large-en-v1.5), (4) GTE (gte-large), (5) ColBERTv2, (6) coCondenser and (7) DPR. 

We also adapted the classic relevance feedback method — the Rocchio algorithm — for the DSE task to assess its effectiveness in this context.

For **reranking**, we have evaluated four models: (1) Stella (stella_en_1.5B_v5), (2) SFR (SFR-Embedding-Mistral), (3) BGE-reranker (bge-reranker-v2-minicpm-layerwise), and (4) LLM (GLM-4-Plus).

The complete evaluation results are available in [./Baselines/evaluation_results.md](./Baselines/evaluation_results.md). For detailed experimental setups and analysis, please refer to the corresponding section in our paper.

The "[./Baselines/Retrieval](./Baselines/Retrieval)" folder and the "[./Baselines/Reranking](./Baselines/Reranking)" folder contain the results for each baseline method, formatted as: `{case_id: {candidate_dataset_id: score, ...}, ...}`.

Below is an example of how to load and process the retrieval results from the `./Baselines/Retrieval/BM25_results.json` file:

```python
import json

# Load the BM25 retrieval result file
with open('Baselines/Retrieval/BM25_results.json', 'r') as f:
    results = json.load(f)
    
    # Iterate through the results for each case
    for case_id, ranking_dict in results.items():
        for candidate_dataset_id, ranking_score in ranking_dict.items():
            print(f"Case ID: {case_id}, Candidate Dataset ID: {candidate_dataset_id}, Score: {ranking_score}")

            # Additional processing for the results...
```

### Baselines for Explanation

We employed post-hoc explanation methods to identify which fields of the candidate dataset are relevant to the query or similar to the target datasets. We evaluated four different explainers: (1) Feature Ablation explainer, (2) LIME explainer, (3) SHAP explainer, and (4) LLM explainer, using F1-score. The first three methods are combined with retrieval models, while the LLM explainer operates independently.

The results for each explainer are stored in the "[./Baselines/Explanation](./Baselines/Explanation)" folder, formatted as: `{case_id: {candidate_dataset_id: {explanaion_type: [0,1,1,0,0], ...}, ...}, ...}`.

The complete evaluation results are available in [./Baselines/evaluation_results.md](./Baselines/evaluation_results.md). For detailed experimental setups and analysis, please refer to the corresponding section in our paper.

Below is an example of how to load and process the retrieval results from the `./Baselines/Explanation/FeatureAblation/BM25_result.json` file:

```python
import json

fields = ['title', 'description', 'tags', 'author', 'summary']

# Load the Feature Ablation explainer combined with BM25 retriever results
with open('Baselines/Explanation/FeatureAblation/BM25_result.json', 'r') as f:
    results = json.load(f)
    
    # Iterate through the results for each case
    for case_id, explanation_dict in results.items():
        for candidate_dataset_id, sub_explanation_dict in explanation_dict.items():
            print(f"Case ID: {case_id}, Candidate Dataset ID: {candidate_dataset_id}")

            # Display field-level query relevance
            print("Field-level Query Relevance:")
            field_level_query_rel = sub_explanation_dict['query']
            for field_idx, field in enumerate(fields):
                if field_level_query_rel[field_idx] == 1:
                    print(f"{field} field is relevant to the query")
                elif field_level_query_rel[field_idx] == 0:
                    print(f"{field} field is irrelevant to the query")

            # Display field-level target similarity
            print("Field-level Target Similarity:")
            field_level_target_sim = sub_explanation_dict['dataset']
            # Additional processing for target similarity...

```


## Source Codes

All implementation source code is available in the [./Code](./Code) directory.

### Dependencies

To run the code, ensure you have the following dependencies installed:

- Python 3.9
- rank-bm25
- scikit-learn
- sentence-transformers
- faiss-gpu
- ragatouille
- tevatron
- torch
- shap
- lime
- zhipuai
- FlagEmbedding

### Retrieval Models

Detailed documentation and code examples for retrieval models are provided in the [./Code/Retrieval/README.md](./Code/Retrieval/README.md).

The retrieval models include:
- **Sparse Retrieval Models:**
  - BM25
  - TF-IDF
- **Dense Retrieval Models:**
  - **Unspervised Dense Retrieval Models:**
    - BGE (bge-large-en-v1.5)
    - GTE (gte-large)
  - **Spervised Dense Retrieval Models:** 
    - coCondenser
    - ColBERTv2
    - DPR

The relevance feedback methods include:
- Rocchio-P
- Rocchio-PN

### Reranking Models

Documentation and code examples for reranking models are provided in the [./Code/Reranking/README.md](./Code/Reranking/README.md).

The reranking models include:
- Stella
- SFR-Embedding-Mistral
- BGE-reranker
- LLM

### Explanation Methods

Documentation and code examples for explanation methods are provided in the [./Code/Explanation/README.md](./Code/Explanation/README.md).

The explanation methods include:
- Feature Ablation
- LIME
- SHAP
- LLM

## LLM Prompts

All prompts are located in [./Code/llm_prompts.py](./Code/llm_prompts.py).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.
