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

# # prepare corpus data
# python build_corpus.py --tokenizer_name "$TOKENIZER" --save_to $DATA_SAVE_TO --truncate $TRUNCATE_LEN

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

# fine-tuning stage 1
bash fine_tuning.sh ./retriever_model_s1 "$CONDENSER_MODEL_NAME" "$DATA_SAVE_TO"/train "$LEARNING_RATE" "$BATCH_SIZE" "$TRUNCATE_LEN"

# prepare hn data
bash encode_and_search.sh \
  "$DATA_SAVE_TO"/corpus \
  "$DATA_SAVE_TO"/query/train.query.json \
  "$ENCODING_SAVE_TO"/corpus \
  "$ENCODING_SAVE_TO"/query/train.pt \
  ./retriever_model_s1 \
  "$TRUNCATE_LEN" \
  train.rank.tsv \
  1000
python build_hn.py --hn_file train.rank.tsv --save_to "$DATA_SAVE_TO" --truncate "$TRUNCATE_LEN" --qrels "$QRELS"

# link
ln -s "$PWD"/"$DATA_SAVE_TO"/train/* "$PWD"/"$DATA_SAVE_TO"/train-hn

# fine-tuning stage 2
bash fine_tuning.sh ./retriever_model_s2 "$CONDENSER_MODEL_NAME" "$DATA_SAVE_TO"/train-hn "$LEARNING_RATE" "$BATCH_SIZE" "$TRUNCATE_LEN"

# encode and search dev set with fine-tuing model
bash encode_and_search.sh \
  "$DATA_SAVE_TO"/corpus \
  "$DATA_SAVE_TO"/query/dev.query.json \
  "$ENCODING_SAVE_TO"/corpus-s2 \
  "$ENCODING_SAVE_TO"/query-s2/qry.pt \
  ./retriever_model_s2 \
  "$TRUNCATE_LEN" \
  dev.rank.tsv \
  "$TOP_K"

# # # evaluate test set
# # if [ "$MERGE" -eq 1 ]; then
# #     python evaluate.py --qrels_file "$QRELS"/test.json --rank_file dev.rank.tsv
# # fi
