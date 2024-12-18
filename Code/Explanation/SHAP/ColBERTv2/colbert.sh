#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python colbert_shap.py  --model_path=./checkpoints/colbert/ \
                                               --mode=qd \
                                               --rel=qexp \
                                               --output_dir=./output \
                                               --annotation=./datasets/faery/human_annotated_qrels.json \
                                               --dataset_info=./datasets/faery/datasets.json \
                                               --queries=./datasets/faery/queries.tsv \
                                               --pooling_path=./shap_lime/results/ColBERTv2.json