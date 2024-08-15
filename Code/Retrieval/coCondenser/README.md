# ColBERTv2

## Installation

We reuse the implementation of [Tevatron](https://github.com/texttron/tevatron/tree/tevatron-v1).

Install Tevatron with pip:

```
git clone -b tevatron-v1 https://github.com/texttron/tevatron
cd tevatron
pip install --editable .
```

## Training

Firstly, prepare a number of data folders named `fold_i`, each data folder includes three files: `train.json`, `valid.json`, `test.json`. Each file is a JSON file, where each JSON object is formatted as: `{qdpair_id: {dataset_id: rel_score, ...}, ...}`. 

Next, prepare `query_info.json` and `dataset_info.json`, where each JSON object is formatted as: `{id: info_text, ...}`. 

Then, run commands as follow:

```
bash pipeline.sh ./bert ./encoding fold_0 1e-5 8 0 20 

```

The parameters are: the directory where the processed data is stored, the directory where the embedded data is stored, the data file directory, the learning rate, the batch size, whether to merge the training and validation sets, and keep the top_k search results. The final results are stored in the `dev.rank.tsv`.