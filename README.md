# DSEBench

DSEBench is a test collection for Dataset Search with Examples (DSE), a task aimed at returning top-ranked candidate datasets based on both a textual query and a set of target datasets that clarify the user's search intent. Unlike traditional dataset retrieval tasks that only use a query, DSE incorporates example datasets to improve search relevance.  We implement experiments on DSE and field-level experiments. For details about this test collection, please refer to the following paper.


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

## Keyword Queries

The "[./Data/queries.tsv](./Data/queries.tsv)" file provides 3,979 keyword queries. Each row represents a query with two "\t"-separated columns: `query_id` and `query_text`. The queries can be divided into generated queries created from the metadata of datasets and NTCIR queries imported from the English part of NTCIR. The IDs of generated queries start with "GEN_", which are used in LLM annotations, while IDs starting with "NTCIR_1" are NTCIR queries used in LLM annotations, and IDs starting with "NTCIR_2" are NTCIR queries used in human annotations.

## Qrels

The "[./Data/human_annotated_qrels.json](./Data/human_annotated_qrels.json)" file contains 7,415 qrels, and the "[./Data/llm_annotated_qrels.json](./Data/llm_annotated_qrels.json)" file contains 122,585 qrels. Each JSON object has eight keys: `query_id`, `target_dataset_id`, `candidate_dataset_id`, `qdpair_id` (the ID of the query-target dataset pair), `qrel` (relevance of a candidate dataset to a query, 0: irrelevant; 1: partially relevant; 2: highly relevant), `query_explanation`, `drel` (relevance of a candidate dataset to a target dataset, 0: irrelevant; 1: partially relevant; 2: highly relevant), and `dataset_explanation`. The `query_explanation` and `dataset_explanation` are both lists of length 5 consisting of 0 and 1, and the order of the corresponding fields is `[title, description, tags, author, summary]`.


```json
{
    "query_id": "NTCIR_200000", 
    "target_dataset_id": "002ece58-9603-43f1-8e2e-54e3d9649e84", 
    "candidate_dataset_id": "99e3b6a2-d097-463f-b6e1-3caceff300c9", 
    "qdpair_id": "1", 
    "qrel": 1, 
    "query_explanation": [1, 1, 1, 0, 0], 
    "drel": 2, 
    "dataset_explanation": [1, 1, 1, 1, 1]
}
```

## Splits for Training, Validation, and Test Sets

To ensure that evaluation results are comparable, one should use the train-validation-test splits that we provide. There are two ways for splitting the data into training, validation, and test sets. The "[./Data/Splits/5-Fold_split](./Data/Splits/5-Fold_split)" folder contains five sub-folders. Each sub-folder provides three qrel files for training, validation, and test sets, respectively. The "[./Data/Splits/Annotators_split](./Data/Splits/Annotators_split)" folder contains three qrel files for training, validation, and test sets, respectively.


## Baselines for DSE

We have evaluated two sparse retrieval models: (1) TF-IDF based cosine similarity, (2) BM25 and five dense retrieval models: (3) BGE, (4) GTE, (5) Contextualized late interaction over BERT (ColBERTv2), (6) coCondenser and (7) Dense Passage Retrieval (DPR). For reranking, we have evaluated three models: (1) Stella, (2) SFR-Embedding-Mistral, (3) GLM-4-Long, and (4) GLM-4-Air.

The details of the experiments are given in the corresponding section of our paper.

The "[./Baselines](./Baselines)" folder provides the results of each baseline method, where each JSON object is formatted as: `{qdpair_id: {dataset_id: score, ...}, ...}`.

## Baselines for Field Relevance

We employed post-hoc explanation methods to identify which fields of the candidate dataset are relevant to the query or target dataset.
We have evaluated four different explainers, (1) feature ablation explainer, (2) LIME, (3) SHAP, (4) LLM, using F1-score, and the first three methods need to be combined with the retrieval models. 

The "[./Baselines](./Baselines)" folder provides the results of each explainers, where each JSON object is formatted as: `{qdpair_id: {dataset_id: {explanaion_type: [0,1,1,0,0], ...}, ...}, ...}`.

For specific experimental details and data, please refer to our paper.


## Source Codes

All source codes of our implementation are provided in [./Code](./Code).

### Dependencies

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

### Sparse Retrieval Models

See codes in [./Code/Retrieval/sparse.py](./Code/Retrieval/sparse.py) for details.

### Dense Retrieval Models

#### Unspervised Models

See codes in [./Code/Retrieval/unsupervised_dense.py](./Code/Retrieval/unsupervised_dense.py) for details.

#### Spervised Models

- **DPR**: See [./Code/Retrieval/DPR](./Code/Retrieval/DPR) for details.
- **coCondenser**: See [./Code/Retrieval/coCondenser](./Code/Retrieval/coCondenser) for details.
- **ColBERTv2**: See [./Code/Retrieval/ColBERTv2](./Code/Retrieval/ColBERTv2) for details.

### Rerank Models

- **BGE-reranker**: See codes in [./Code/Rerank/bge_reranker.py](./Code/Rerank/bge_reranker.py) for details.
- **Stella & SFR**: See codes in [./Code/Rerank/dense_reranker.py](./Code/Rerank/dense_reranker.py) for details.
- **LLM**: See [./Code/Retrieval/LLM](./Code/Retrieval/LLM) for details.

### Explanation Methods

- **Feature Ablation**: See codes in [./Code/Explanation/feature_ablation.py](./Code/Explanation/feature_ablation.py) for details.
- **SHAP**: See [./Code/Explanation/SHAP](./Code/Explanation/SHAP) for details.
- **LIME**: See [./Code/Explanation/LIME](./Code/Explanation/LIME) for details.
- **LLM**: See [./Code/Explanation/LLM](./Code/Explanation/LLM) for details.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

