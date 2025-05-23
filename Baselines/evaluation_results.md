# Evaluation Results of Retrieval Models

|                      | MAP@5    | @10      | NDCG@5   | @10      | R@5     | @10     |
| :------------------- | :------- | :------- | :------- | :------- | :------ | :------ |
| *Unsupervised Models* |          |          |          |          |         |         |
| BM25                 | 0.0982   | 0.1739   | 0.3059   | 0.3416   | 0.1705  | 0.2769  |
| TF-IDF               | 0.0921   | 0.1615   | 0.2971   | 0.3227   | 0.1572  | 0.2576  |
| BGE                  | 0.1045   | 0.1834   | 0.3233   | 0.3598   | 0.1756  | 0.2887  |
| GTE                  | 0.1104   | 0.1921   | 0.3267   | 0.3649   | 0.1820  | 0.2983  |
| *Supervised Models (not fine-tuned)* | | | | | | |
| DPR                  | 0.1072   | 0.1757   | 0.3248   | 0.3472   | 0.1706  | 0.2695  |
| ColBERTv2            | 0.1072   | 0.1799   | 0.3206   | 0.3510   | 0.1687  | 0.2720  |
| coCondenser          | 0.1029   | 0.1671   | 0.3142   | 0.3326   | 0.1597  | 0.2525  |
| *Supervised Models (fine-tuned with the five-fold split)* | | | | | | |
| DPR                  | 0.1204   | 0.1929   | 0.3574   | 0.3769   | 0.1833  | 0.2967  |
| ColBERTv2            | 0.1247   | 0.2052   | 0.3449   | 0.3779   | 0.1909  | 0.3066  |
| coCondenser          | **0.1387** | **0.2286** | **0.3784** | **0.4147** | **0.2110** | **0.3401** |
| *Supervised Models (fine-tuned with the annotator split)* | | | | | | |
| DPR                  | 0.1194   | 0.1936   | 0.3551   | 0.3757   | 0.1851  | 0.2914  |
| ColBERTv2            | 0.1109   | 0.1925   | 0.3360   | 0.3760   | 0.1800  | 0.3033  |
| coCondenser          | 0.1286   | 0.2238   | 0.3656   | 0.4136   | 0.1979  | 0.3367  |
| *RF Methods* |          |          |          |          |         |         |
| Rocchio-P            | 0.1033   | 0.1782   | 0.3139   | 0.3452   | 0.1713  | 0.2837  |
| Rocchio-PN           | 0.1023   | 0.1773   | 0.3125   | 0.3441   | 0.1696  | 0.2828  |


# Evaluation Results of Reranking Models

|                      | MAP@5    | @10      | NDCG@5   | @10      | R@5     | @10     |
| :------------------- | :------- | :------- | :------- | :------- | :------ | :------ |
| *Supervised Models (not fine-tuned)* |          |          |          |          |         |         |
| Stella               | 0.1180   | 0.2106   | 0.3509   | 0.3981   | 0.1938  | 0.3307  |
| SFR                  | 0.1184   | 0.2090   | 0.3488   | 0.3920   | 0.1940  | 0.3225  |
| BGE-reranker         | 0.1170   | 0.2032   | 0.3470   | 0.3877   | 0.1930  | 0.3163  |
| *Supervised Models (fine-tuned with the five-fold split)* | | | | | | |
| BGE-reranker         | 0.1178   | 0.2085   | 0.3464   | 0.3956   | 0.1914  | 0.3273  |
| *Supervised Models (fine-tuned with the annotator split)* | | | | | | |
| BGE-reranker         | 0.1249   | 0.2146   | 0.3665   | 0.4099   | 0.1985  | 0.3347  |
| *LLM* |          |          |          |          |         |         |
| zero-shot            | 0.1144   | 0.1748   | 0.3130   | 0.3360   | 0.1691  | 0.2652  |
| one-shot             | 0.1154   | 0.1776   | 0.3058   | 0.3327   | 0.1691  | 0.2729  |
| multi-layer          | **0.1468** | **0.2398** | **0.4071** | **0.4451** | **0.2093** | **0.3608** |


# Evaluation Results (F1) of Explanation Methods

| Retrieval Model                                         | Explanation Method | Indicator Fields for Query Relevance | Indicator Fields for Target Similarity |
| :------------------------------------------------------ | :----------------- | :----------------------------------- | :------------------------------------- |
| BM25                                                    | Ablation           | 0.4819                               | 0.4750                                 |
|                                                         | LIME               | 0.6325                               | 0.7562                                 |
|                                                         | SHAP               | 0.6564                               | **0.8177** |
|                                                         | LLM (zero-shot)    | 0.6455                               | 0.7315                                 |
|                                                         | LLM (few-shot)     | **0.7246** | 0.7647                                 |
| TF-IDF                                                  | Ablation           | 0.4199                               | 0.4174                                 |
|                                                         | LIME               | 0.6504                               | 0.7398                                 |
|                                                         | SHAP               | 0.6573                               | **0.8172** |
|                                                         | LLM (zero-shot)    | 0.6491                               | 0.7299                                 |
|                                                         | LLM (few-shot)     | **0.7319** | 0.7636                                 |
| BGE                                                     | Ablation           | 0.4656                               | 0.3463                                 |
|                                                         | LIME               | 0.6736                               | 0.7846                                 |
|                                                         | SHAP               | 0.6634                               | **0.8080** |
|                                                         | LLM (zero-shot)    | 0.6413                               | 0.7263                                 |
|                                                         | LLM (few-shot)     | **0.7251** | 0.7651                                 |
| GTE                                                     | Ablation           | 0.4277                               | 0.3453                                 |
|                                                         | LIME               | 0.6713                               | 0.7881                                 |
|                                                         | SHAP               | 0.6789                               | **0.8018** |
|                                                         | LLM (zero-shot)    | 0.6354                               | 0.7267                                 |
|                                                         | LLM (few-shot)     | **0.7176** | 0.7627                                 |
| DPR <br/> (not fine-tuned)                              | Ablation           | 0.4282                               | 0.3296                                 |
|                                                         | LIME               | 0.6992                               | 0.7650                                 |
|                                                         | SHAP               | 0.6718                               | **0.8201** |
|                                                         | LLM (zero-shot)    | 0.6492                               | 0.7450                                 |
|                                                         | LLM (few-shot)     | **0.7316** | 0.7757                                 |
| ColBERTv2 <br/> (not fine-tuned)                        | Ablation           | 0.4849                               | 0.3825                                 |
|                                                         | LIME               | 0.6868                               | 0.7659                                 |
|                                                         | SHAP               | 0.6662                               | **0.8154** |
|                                                         | LLM (zero-shot)    | 0.6398                               | 0.7423                                 |
|                                                         | LLM (few-shot)     | **0.7248** | 0.7667                                 |
| coCondenser <br/> (not fine-tuned)                      | Ablation           | 0.4133                               | 0.3195                                 |
|                                                         | LIME               | 0.6897                               | 0.7607                                 |
|                                                         | SHAP               | 0.6406                               | **0.8217** |
|                                                         | LLM (zero-shot)    | 0.6409                               | 0.7444                                 |
|                                                         | LLM (few-shot)     | **0.7161** | 0.7690                                 |
| DPR <br/> (fine-tuned with<br/> the five-fold split)    | Ablation           | 0.4393                               | 0.3641                                 |
|                                                         | LIME               | 0.6841                               | 0.7840                                 |
|                                                         | SHAP               | 0.6652                               | **0.8069** |
|                                                         | LLM (zero-shot)    | 0.6483                               | 0.7345                                 |
|                                                         | LLM (few-shot)     | **0.7423** | 0.7715                                 |
| ColBERTv2 <br/> (fine-tuned with<br/> the five-fold split) | Ablation           | 0.4020                               | 0.3327                                 |
|                                                         | LIME               | 0.6845                               | 0.7788                                 |
|                                                         | SHAP               | 0.6741                               | **0.7943** |
|                                                         | LLM (zero-shot)    | 0.6372                               | 0.7354                                 |
|                                                         | LLM (few-shot)     | **0.7246** | 0.7647                                 |
| coCondenser <br/> (fine-tuned with<br/> the five-fold split) | Ablation           | 0.4193                               | 0.3351                                 |
|                                                         | LIME               | 0.6955                               | 0.7766                                 |
|                                                         | SHAP               | 0.6761                               | **0.8011** |
|                                                         | LLM (zero-shot)    | 0.6351                               | 0.7264                                 |
|                                                         | LLM (few-shot)     | **0.7186** | 0.7638                                 |
| DPR <br/> (fine-tuned with <br/> the annotator split)   | Ablation           | 0.5202                               | 0.4038                                 |
|                                                         | LIME               | 0.7288                               | 0.7616                                 |
|                                                         | SHAP               | 0.6800                               | **0.8023** |
|                                                         | LLM (zero-shot)    | 0.6437                               | 0.7222                                 |
|                                                         | LLM (few-shot)     | **0.7337** | 0.7629                                 |
| ColBERTv2 <br/> (fine-tuned with <br/> the annotator split) | Ablation           | 0.5047                               | 0.4045                                 |
|                                                         | LIME               | 0.6798                               | 0.7801                                 |
|                                                         | SHAP               | 0.6844                               | **0.7965** |
|                                                         | LLM (zero-shot)    | 0.6351                               | 0.7236                                 |
|                                                         | LLM (few-shot)     | **0.7260** | 0.7644                                 |
| coCondenser <br/> (fine-tuned with<br/> the annotator split) | Ablation           | 0.4754                               | 0.3675                                 |
|                                                         | LIME               | 0.7100                               | 0.7529                                 |
|                                                         | SHAP               | 0.6894                               | **0.7899** |
|                                                         | LLM (zero-shot)    | 0.6334                               | 0.7121                                 |
|                                                         | LLM (few-shot)     | **0.7239** | 0.7508                                 |

