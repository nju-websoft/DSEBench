#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python bge_lime.py  --model_path=../../../../model/bge-large-en-v1.5 \
                                           --mode=qd \
                                           --rel=qexp \
                                           --output_dir=./output \
                                           --annotation=../../../../datasets/faery/human_annotation_qrels.json \
                                           --dataset_info=../../../../datasets/faery/datasets.json \
                                           --queries=../../../../datasets/faery/queries.tsv \
                                           --pooling_path=../../../shap_lime/results/bge_results.json