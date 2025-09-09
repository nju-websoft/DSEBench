import pytrec_eval
from rank_llm.data import Query, Candidate, Request
from rank_llm.rerank import Reranker
from rank_llm.rerank.listwise import SafeOpenai
from rank_llm.rerank.listwise.listwise_rankllm import PromptMode
import json

zhipu_api_key = ''
llm_model_name = "GLM-4-Plus"


with open('rerank_10.json', 'r') as f:
    rerank_dict = json.load(f)

dataset_info_dict = {}
with open('dataset_metadata.json', 'r') as f:
    data = json.load(f)
    for item in data:
        dataset_id, title, description, tags, author, summary = item
        dataset_info_dict[dataset_id] = f'Title: {title}\nDescription: {description}\nTags: {tags}\nAuthor: {author}\nSummary: {summary}'

query_info_dict = {}
with open('query_info.json', 'r') as f:
    data = json.load(f)
    for item in data:
        query_info_dict[item['id']] = f"- Keyword Query: {item['keywords']}\n- Target Dataset:\n{dataset_info_dict[item['dataset_id']]}"

with open('bm25_scores.json', 'r') as f:
    bm25_scores = json.load(f)

retrieval_res = {}
for query_id, doc_ids in rerank_dict.items():
    retrieval_res[query_id] = {}
    for doc_id in doc_ids:
        retrieval_res[query_id][doc_id] = bm25_scores[query_id][doc_id]

query_texts, doc_texts = query_info_dict, dataset_info_dict

requests = []
for query_id, doc_scores in retrieval_res.items():
    candidates = [Candidate(docid=doc_id, score=score, doc={"contents": doc_texts[doc_id]}) for doc_id, score in doc_scores.items()]
    candidates.sort(key=lambda x: x.score, reverse=True)
    requests.append(Request(query=Query(text=query_texts[query_id], qid=query_id), candidates=candidates))

try:
    agent = SafeOpenai(
        model=llm_model_name,
        context_size=32000,
        keys=zhipu_api_key,
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        # prompt_mode=PromptMode.RANK_GPT,
        prompt_template_path="rank_DSE_template.yaml",
        window_size=20
    )
    reranker = Reranker(agent)
    rerank_results = reranker.rerank_batch(requests=requests)

    rerank_scores_dict = {}
    rerank_rank_dict = {}
    for result in rerank_results:
        query_id = result.query.qid
        rerank_scores_dict[query_id] = {}
        rerank_rank_dict[query_id] = {}
        # print(f"Query: {result.query.text}")
        for i, cand in enumerate(result.candidates):
            doc_id = cand.docid
            rerank_scores_dict[query_id][doc_id] = cand.score
            rerank_rank_dict[query_id][doc_id] = i
            # print(f"  Rank {i+1}: docid={cand.docid}, New Score={cand.score:.4f}, content='{cand.doc['contents'][:60]}...'")
        # print("-" * 20)

    with open(f'{llm_model_name.lower()}_rank-llm.json', 'w') as f:
        json.dump(rerank_scores_dict, f, indent=2)
    with open(f"{llm_model_name.lower()}_rank-llm_rank.json", 'w') as f:
        json.dump(rerank_rank_dict, f, indent=2)

except Exception as e:
    print(f"An error occurred: {e}")


def read_qrels_origin(filename='qrels.txt'):
    qrels_kw, qrels_ds = {}, {}
    with open(filename, 'r') as f:
        for line in f:
            if not line:
                continue
            data = line.split('\t')
            query_id, dataset_id, rel_k, rel_d = data[0], data[3], data[4], data[5]
            if query_id not in qrels_kw:
                qrels_kw[query_id] = {}
            qrels_kw[query_id][dataset_id] = int(rel_k)
            if query_id not in qrels_ds:
                qrels_ds[query_id] = {}
            qrels_ds[query_id][dataset_id] = int(rel_d)
    return qrels_kw, qrels_ds

def get_qrels_multi(qrels_kw, qrels_ds):
    qrels_multi = {}
    for query_id, rel_dict in qrels_kw.items():
        qrels_multi[query_id] = {}
        for dataset_id, rel in rel_dict.items():
            qrels_multi[query_id][dataset_id] = int(rel) * int(qrels_ds[query_id][dataset_id])
    return qrels_multi

def output_eval_result(qrels_dict, run_dict, metrics):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_dict, metrics)
    eval_results = evaluator.evaluate(run_dict)
    results = {}
    for metric in metrics:
        results[metric] = sum([x[metric] for x in eval_results.values()]) / len(eval_results)
    for metric in metrics:
        print(f'{metric}: {results[metric]:.4f}', end='\t')
    print()

qrels_kw, qrels_ds = read_qrels_origin()
qrels_dict = get_qrels_multi(qrels_kw, qrels_ds)
metrics = ['map_cut_5', 'ndcg_cut_5', 'P_5', 'recall_5', 'map_cut_10', 'ndcg_cut_10', 'P_10', 'recall_10']
output_eval_result(qrels_dict, rerank_scores_dict, metrics)
