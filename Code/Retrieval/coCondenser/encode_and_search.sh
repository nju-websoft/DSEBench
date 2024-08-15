CORPUS_IN=$1
QUERY_IN=$2
CORPUS_OUT=$3
QUERY_OUT=$4
MODEL_NAME=$5
MAX_LEN=$6
RANK_FILE=$7
SEARCH_DEPTH=$8

BATCH_SIZE=128

mkdir -p "$CORPUS_OUT"
mkdir -p "$(dirname "$QUERY_OUT")"


# 遍历目录中的所有文件
for FILE in "$CORPUS_IN"/*.json; do
  # 获取文件名（不包括路径）
  FILENAME=$(basename "$FILE")
  
  # 去掉文件名的后缀 .json
  BASENAME="${FILENAME%.json}"
  
  # 提取文件名去掉后缀后的最后两位字符
  LAST_TWO_CHARS="${BASENAME: -2}"
  
  python -m tevatron.driver.encode \
  --output_dir ./retriever_model \
  --model_name_or_path "$MODEL_NAME" \
  --fp16 \
  --p_max_len "$MAX_LEN" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --encode_in_path "$CORPUS_IN"/split"$LAST_TWO_CHARS".json \
  --encoded_save_path "$CORPUS_OUT"/split"$LAST_TWO_CHARS".pt
done

python -m tevatron.driver.encode \
  --output_dir ./retriever_model \
  --model_name_or_path "$MODEL_NAME" \
  --fp16 \
  --q_max_len "$MAX_LEN" \
  --encode_is_qry \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --encode_in_path "$QUERY_IN" \
  --encoded_save_path "$QUERY_OUT"

python -m tevatron.faiss_retriever \
--query_reps "$QUERY_OUT" \
--passage_reps "$CORPUS_OUT"/'*.pt' \
--depth "$SEARCH_DEPTH" \
--batch_size -1 \
--save_text \
--save_ranking_to "$RANK_FILE"