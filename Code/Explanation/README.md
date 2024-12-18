# LIME
- lime 0.2.0.1

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
```--model_path```: the path or name of retrieval model. For BM25 and TF-IDF, this parameter is not required.

```--mode```: "qd" means the input of the model is the concatenation of query and target dataset; "q" means the input of the model is the query; "d" means the input of the model is the target dataset.

```--rel```: "qexp" means the explanation of query relevance; "dexp" means the explanation of dataset relevance.

```--output_dir```: the directory of output file

```--annotation```: the path of human or llm annotated file

```--dataset_info```: the path of datasets.json

```--queries```: the path of queries.tsv

```--pooling_path```: the path of retrieved result

# SHAP

- shap  0.46.0

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

The meaning of the parameters is the same as above.

# LLM

```bash
python zhipu.py --mode=qd \
                --rel=qexp \
                --output_dir=./output \
                --annotation=./datasets/faery/human_annotation_qrels.json \
                --dataset_info=./datasets/faery/datasets.json \
                --queries=./datasets/faery/queries.tsv
```
The meaning of the parameters is the same as above.