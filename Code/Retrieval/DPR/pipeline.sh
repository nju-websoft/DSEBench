# ./bert
DATA_SAVE_TO=$1
# ./encoding
ENCODING_SAVE_TO=$2
QRELS=$3

# 5e-6
LEARNING_RATE=$4
# 8
BATCH_SIZE=$5
# 0: grid search, 1: train
MERGE=$6
# 10/20
TOP_K=$7

TRUNCATE_LEN=512
CONDENSER_MODEL_NAME=Luyu/co-condenser-marco-retriever
TOKENIZER=bert-base-uncased

# prepare corpus data
python build_corpus.py --tokenizer_name "$TOKENIZER" --save_to $DATA_SAVE_TO --truncate $TRUNCATE_LEN

# prepare query data
python build_query.py --save_to "$DATA_SAVE_TO" --qrels "$QRELS" --merge_valid "$MERGE"

# encode and search train set with coCondenser model to supplement negatives
bash encode_and_search.sh \
  "$DATA_SAVE_TO"/corpus \
  "$DATA_SAVE_TO"/query/train.query.json \
  "$ENCODING_SAVE_TO"/corpus \
  "$ENCODING_SAVE_TO"/query/train.pt \
  "$CONDENSER_MODEL_NAME" \
  "$TRUNCATE_LEN" \
  train.rank.tsv \
  100

# prepare train data
python build_train.py --save_to "$DATA_SAVE_TO" --qrels "$QRELS" --merge_valid "$MERGE" --hn_file train.rank.tsv

# training
python -m tevatron.driver.train \
  --output_dir ./dpr_model \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --train_dir "$DATA_SAVE_TO"/train \
  --fp16 \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --positive_passage_no_shuffle \
  --train_n_passages 2 \
  --learning_rate "$LEARNING_RATE" \
  --q_max_len "$TRUNCATE_LEN" \
  --p_max_len "$TRUNCATE_LEN" \
  --num_train_epochs 40 \
  --logging_steps 500 \
  --overwrite_output_dir

# encode and search dev set with fine-tuing model
bash encode_and_search.sh \
  "$DATA_SAVE_TO"/corpus \
  "$DATA_SAVE_TO"/query/dev.query.json \
  "$ENCODING_SAVE_TO"/corpus-s2 \
  "$ENCODING_SAVE_TO"/query-s2/qry.pt \
  ./dpr_model \
  512 \
  dev.rank.tsv \
  "$TOP_K"