# Explanation Methods

This folder contains four explanation methods:

- Feature Ablation
- LIME
- SHAP
- LLM

## Feature Ablation

The Feature Ablation method masks out different parts of dataset information to determine their contribution to the model's relevance score.

Before using feature ablation, prepare a `dataset_mask_info.json` file. The format should be:

```json
{
    "dataset_id_1": {
        "title": "dataset_info_mask_title",
        "description": "dataset_info_mask_description",
        "tags": "dataset_info_mask_tags",
        "author": "dataset_info_mask_author",
        "summary": "dataset_info_mask_summary",
        "full": "dataset_info"
    },
    "dataset_id_2": { ... }
    ...
}
```

For more details on how to run and use feature ablation, refer to [./feature_ablation.py](./feature_ablation.py).

## LIME

LIME version `0.2.0.1` is required for this method.

For different retrieval model, an example of command as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python bge_lime.py  --model_path=BAAI/bge-large-en-v1.5 \
                                           --mode=qd \
                                           --rel=qexp \
                                           --output_dir=./output \
                                           --annotation=./datasets/faery/human_annotation_qrels.json \
                                           --dataset_info=./datasets/faery/datasets.json \
                                           --queries=./datasets/faery/queries.tsv \
                                           --pooling_path=./results/bge_results.json
```

Parameters:

- `--model_path`: Path or name of the retrieval model (not required for BM25 and TF-IDF).
- `--mode`: Mode of input (qd for query + target dataset, q for query, d for dataset).
- `--rel`: Explanation type (qexp for query relevance, dexp for dataset relevance).
- `--output_dir`: Directory for output files.
- `--annotation`: Path to the human or LLM annotated file.
- `--dataset_info`: Path to the datasets.json file.
- `--queries`: Path to the queries.tsv file.
- `--pooling_path`: Path to the retrieved result.

## SHAP

SHAP version `0.46.0` is required for this method.

For different retrieval model, an example of command as follows:

```bash
CUDA_VISIBLE_DEVICES=0 python gte_shap.py  --model_path=thenlper/gte-large \
                                           --mode=qd \
                                           --rel=qexp \
                                           --output_dir=./output \
                                           --annotation=./datasets/faery/human_annotation_qrels.json \
                                           --dataset_info=./datasets/faery/datasets.json \
                                           --queries=./datasets/faery/queries.tsv \
                                           --pooling_path=./shap_lime/results/gte_results.json
```

The parameters have the same meanings as described in the LIME section above.


## LLM

For LLM-based explanations, use the following command:

```bash
python zhipu.py --mode=qd \
                --rel=qexp \
                --output_dir=./output \
                --annotation=./datasets/faery/human_annotation_qrels.json \
                --dataset_info=./datasets/faery/datasets.json \
                --queries=./datasets/faery/queries.tsv
```

The parameters for this command are similar to those used in the LIME and SHAP examples.
