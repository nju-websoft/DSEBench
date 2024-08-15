OUTPUT_DIR=$1
CONDENSER_MODEL_NAME=$2
TRAIN_DIR=$3

LR=$4
BATCH_SIZE=$5
MAX_LEN=$6

EPOCH=3

python -m tevatron.driver.train \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$CONDENSER_MODEL_NAME" \
  --save_steps 20000 \
  --train_dir "$TRAIN_DIR" \
  --fp16 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --learning_rate "$LR" \
  --num_train_epochs "$EPOCH" \
  --dataloader_num_workers 2  \
  --q_max_len "$MAX_LEN"  \
  --p_max_len "$MAX_LEN"