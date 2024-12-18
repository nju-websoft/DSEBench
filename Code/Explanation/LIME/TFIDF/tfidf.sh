#!/bin/bash
python tfidf_lime.py --mode=qd \
                     --rel=qexp \
                     --output_dir=./output \
                     --annotation=./datasets/faery/human_annotated_qrels.json \
                     --dataset_info=./datasets/faery/datasets.json \
                     --queries=./datasets/faery/queries.tsv \
                     --pooling_path=./results/tfidf_results.json