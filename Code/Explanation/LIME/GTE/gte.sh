#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python gte_lime.py  --model_path=../../../../model/gte-large \
                                           --mode=qd \
                                           --rel=qexp \
                                           --output_dir=./output \
                                           --annotation=../../../../datasets/faery/human_annotation_qrels.json \
                                           --dataset_info=../../../../datasets/faery/datasets.json \
                                           --queries=../../../../datasets/faery/queries.tsv \
                                           --pooling_path=../../../shap_lime/results/gte_results.json