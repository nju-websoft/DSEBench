#!/bin/bash
python zhipu.py     --mode=qd \
                    --rel=qexp \
                    --output_dir=./output \
                    --annotation=./datasets/faery/human_annotation_qrels.json \
                    --dataset_info=./datasets/faery/datasets.json \
                    --queries=./datasets/faery/queries.tsv
