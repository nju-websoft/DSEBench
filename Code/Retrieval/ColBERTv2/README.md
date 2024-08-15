# ColBERTv2

## Installation

We reuse the implementation of [RAGatouille](https://github.com/AnswerDotAI/RAGatouille).

Install RAGatouille with pip:

```
pip install ragatouille
```

## Training

Firstly, prepare a number of data folders named `fold_i`, each data folder includes three files: `train.json`, `valid.json`, `test.json`. Each file is a JSON file, where each JSON object is formatted as: `{qdpair_id: {dataset_id: rel_score, ...}, ...}`. 

Next, prepare `query_info.json` and `dataset_info.json`, where each JSON object is formatted as: `{id: info_text, ...}`. 

Then, run commands as follow:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python grid_search.py \
--doc_maxlen 512 \
--top_k 20 \
--fold_start 0 \
--fold_end 5 \
--learning_rate 1e-05 \
--batch_size 8 \
--mode train

```

