DATA_SAVE_TO=$1
QRELS=$2

ENCODING_SAVE_TO=$3
MODEL_NAME=$4
TRUNCATE_LEN=512

TOP_K=$5

# prepare query data
python build_query.py --save_to "$DATA_SAVE_TO" --qrels "$QRELS" --merge_valid 1

# encode and search train set with coCondenser model to supplement negatives
bash encode_and_search.sh \
  "$DATA_SAVE_TO"/corpus \
  "$DATA_SAVE_TO"/query/test.query.json \
  "$ENCODING_SAVE_TO"/corpus \
  "$ENCODING_SAVE_TO"/query/qry.pt \
  "$MODEL_NAME" \
  "$TRUNCATE_LEN" \
  test.rank.tsv \
  "$TOP_K"