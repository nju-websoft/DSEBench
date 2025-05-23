import pandas as pd
import json
from tqdm import tqdm


with open('../data/datasets.json') as f:
    datasets = json.load(f)

dataset_ids = []
for dataset in datasets:
    dataset_ids.append(dataset['id'])
queries = pd.read_csv('../data/queries.tsv', sep='\t', header=None)
# print(queries)
queries_dict = {}
for q in queries.iterrows():
    queries_dict[q[1][0]] = q[1][1]

dfq = pd.read_csv('../data/cases.tsv', sep='\t', header=None)
print(dfq)
print(dfq[1].str.contains('NTCIR_2', na=False))
dfq = dfq[dfq[1].str.contains('NTCIR_2', na=False)].copy()

print(dfq)
pair_dict = {}
for i in range(len(dfq)):
    pair_dict[dfq[0][i]] = [dfq[1][i], dfq[2][i]]
print(len(pair_dict), pair_dict['1'])
num_queries = len(dfq)

for i in range(len(dfq)):
    dfq[1][i] = queries_dict[dfq[1][i]]

print(dfq)
# print(datasets[0])

sent = []
for i in range(len(datasets)):
    if (i % 10000 == 0):
        print(i)
    tokens = "\n".join(
        [datasets[i]['title'], datasets[i]['description'], ", ".join(datasets[i]['tags']), datasets[i]['author'],
         datasets[i]['summary']])
    tokens_list = tokens.split()
    sent.append(tokens_list)
print(sent[0])

docs2 = []
docs = sent
for st in docs:
    st = ' '.join(st)
    docs2.append(st)
docs = docs2
docs2 = []
print(docs[0])

sent_q = []
for i in range(len(dfq)):
    tokens = dfq[1][i]
    tokens_list = tokens.split()
    sent_q.append(tokens_list)

print(sent_q[0])

docs2_q = []
docs_q = sent_q
for st in docs_q:
    st = ' '.join(st)
    docs2_q.append(st)
docs_q = docs2_q
docs2_q = []
print(docs_q[0])

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)
print(X.shape)
Y = vectorizer.transform(docs_q)
print(Y.shape)
print(Y[0])

list_q = []
with tqdm(total=len(dfq)) as pbar:
    for i in range(len(dfq)):
        a = Y[i].toarray()
        l = list(a[0])
        list_q.append(l)
        pbar.update(1)

import numpy as np

array_q = np.array(list_q, dtype='float32')
print(array_q.shape)

cos_lib = np.zeros(shape=(len(datasets), len(dfq)))  # to store distances

from sklearn.metrics.pairwise import cosine_similarity

cos_lib = cosine_similarity(Y, X)
print(len(cos_lib[0]))

# print(sorted(cos_lib[0], reverse=True))

dist = []
for i in range(len(cos_lib[0])):
    dist.append((i, cos_lib[0][i]))

dist = sorted(dist, key=lambda x: x[1], reverse=True)
# print(dist[0:5])
with open('../data/cases.tsv', 'r') as f:
    cases = pd.read_csv(f, sep='\t', header=None)

threshold = 0.15
rel_docs = []  # to store relev
actual_rel_docs = []  # actual rel
rel_docs_ids = []  # to store indices of rel doc vectors
p = 0

with tqdm(total=num_queries) as pbar:
    for k in range(num_queries):
        dist = []
        a = []  # to store relev
        b = []  # actual rel
        ai = []  # to store indices of relev
        for i in range(len(cos_lib[k])):
            dist.append((i, cos_lib[k][i]))

        dist = sorted(dist, key=lambda x: x[1], reverse=True)
        i = 0
        while (dist[i][1] >= threshold and i < len(datasets)):
            # print(dist[i]," ",df3["File_Name"][dist[i][0]].split('\\')[-1])
            a.append(datasets[dist[i][0]]['id'])
            ai.append(dist[i][0])
            i = i + 1

        mdf = (cases[cases[0] == dfq[0][k]])
        assert len(mdf) == 1
        for row in mdf.itertuples():
            b.append(row[-1])
            # print(mdf["Doc_Name"][p])
        rel_docs.append(a)
        actual_rel_docs.append(b)  # actual rel
        rel_docs_ids.append(ai)
        pbar.update(1)

rel_docs = actual_rel_docs
matches = 0
# for i in range(num_queries):
#     print(len(actual_rel_docs[i]), " ", len(set(rel_docs[i]) & set(actual_rel_docs[i])))

matches = 0
for i in range(num_queries):
    matches = matches + len(set(rel_docs[i]) & set(actual_rel_docs[i]))

print("Total Matches: ", matches)
import numpy as np

temp = np.zeros((1, X.shape[1]), dtype='float32')
for k in range(len(datasets)):
    temp = np.add(temp, (X[k].toarray()))  # storing relevant vectors

import pandas as pd

import numpy as np

dr_rev = np.zeros((num_queries, X.shape[1]), dtype='float32')
ndr_rev = np.zeros((num_queries, X.shape[1]), dtype='float32')

with tqdm(total=num_queries) as pbar:
    for i in range(num_queries):

        tempnr = np.zeros((1, X.shape[1]), dtype='float32')
        tempr = np.zeros((1, X.shape[1]), dtype='float32')

        matched_files = list(set(rel_docs[i]) & set(actual_rel_docs[i]))
        mat = len(matched_files)
        nmat = len(datasets) - mat
        for k in range(len(datasets)):
            if (datasets[k]['id'] in matched_files):
                tempr = np.add(tempr, (X[k].toarray()))  # storing relevant vectors

        tempnr = np.subtract(temp, tempr)

        if (mat != 0):
            dr_rev[i] = tempr / mat  # finding average of all relevant doc vectors
        if (nmat != 0):
            ndr_rev[i] = tempnr / nmat
        pbar.update(1)

beta = 0.75
gamma = 0
mod_q = np.zeros((num_queries, X.shape[1]), dtype='float32')

with tqdm(total=num_queries) as pbar:
    for i in range(num_queries):
        if (len(list(set(rel_docs[i]) & set(actual_rel_docs[i]))) > 0):  # checking no. of matches > 0 or not
            total_rel = len(list(set(rel_docs[i]) & set(actual_rel_docs[i])))
            total_non_rel = len(datasets) - total_rel
            term = beta
            mod_q[i] = np.add(array_q[i], term * dr_rev[i])
            mod_q[i] = np.subtract(mod_q[i], (gamma / total_non_rel) * ndr_rev[i])
        else:
            total_non_rel = len(datasets)
            mod_q[i] = array_q[i]
            mod_q[i] = np.subtract(mod_q[i], (gamma / total_non_rel) * ndr_rev[i])
        pbar.update(1)

cos_lib2 = cosine_similarity(mod_q, X)
rel_docs2 = []  # to store relev
rel_docs2_score = []
p = 0
result = {}
with tqdm(total=num_queries) as pbar:
    for k in range(num_queries):
        result[dfq[0][k]] = {}
        dist2 = []
        a2 = []  # to store relev
        a2_score = []
        b2 = []  # actual rel
        for i in range(len(cos_lib2[k])):
            dist2.append((i, cos_lib2[k][i]))

        dist2 = sorted(dist2, key=lambda x: x[1], reverse=True)
        i = 0
        # while (dist2[i][1] >= threshold and i < len(datasets)):
        while (len(a2) < 20 and i < len(datasets)):
            # print(dist[i]," ",df3["File_Name"][dist[i][0]].split('\\')[-1])
            a2.append(datasets[dist2[i][0]]['id'])
            a2_score.append(dist2[i][1])
            i = i + 1

        rel_docs2.append(a2)
        for id, score in zip(a2, a2_score):
            result[dfq[0][k]][id] = score
        pbar.update(1)

matches = 0
for i in range(num_queries):
    matches = matches + len(set(rel_docs2[i]) & set(actual_rel_docs[i]))

print("Total Matches: ", matches)
with open('../output/rocchio_gamma0.json', 'w') as f:
    json.dump(result, f, indent=4)
