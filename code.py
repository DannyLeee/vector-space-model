from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
from datetime import datetime,timezone,timedelta

def timestamp():
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(dt2)

timestamp()
doc_path = "./data/docs/"
query_path = "./data/queries/"

d_list = []
with open('./data/doc_list.txt', 'r') as d_list_file:
    for line in d_list_file:
        line = line.replace("\n", "")
        d_list += [line]

q_list = []
with open('./data/query_list.txt', 'r') as q_list_file:
    for line in q_list_file:
        line = line.replace("\n", "")
        q_list += [line]

def file_iter(_type):
    if _type == "q":
        for name in q_list:
            with open(query_path+name+'.txt') as f:
                yield f.readline()
    elif _type == "d":
        for name in d_list:
            with open(doc_path+name+'.txt') as f:
                yield f.readline()

doc_list = []
list_tf = []
for txt in tqdm(file_iter("q")):
    list_tf += [Counter(txt.split())]

for txt in tqdm(file_iter("d")):
    doc_list += [txt]

#tf
df_q_tf = pd.DataFrame(list_tf)
df_q_tf = df_q_tf.fillna(0)
np_q_tf = np.array(df_q_tf)

np_d_tf = np.empty([len(d_list), 123])
for i,t in tqdm(enumerate(doc_list)):
    for j, w in enumerate(df_q_tf.columns):
        np_d_tf[i][j] += t.count(w)
del df_q_tf

# idf
np_idf =  np.log(1+(len(d_list)+1) / (np.count_nonzero(np_d_tf, axis=0)+1))

# tfidf
np_q_tfidf = (1 + np.ma.log2(np_q_tf)) * np_idf
np_d_tfidf = (1 + np.ma.log2(np_d_tf)) * np_idf
del np_q_tf, np_d_tf

np_q_tfidf = np_q_tfidf.filled(0)
np_d_tfidf = np_d_tfidf.filled(0)

# sim_array
sim_array = cosine_similarity(np_q_tfidf, np_d_tfidf)
sim_array = np.array(sim_array)

# output
with open('result.csv', 'w') as output_file:
    output_file.write("Query,RetrievedDocuments\n")
    for i, q_id in tqdm(enumerate(q_list)):
        output_file.write(q_id+',')
        sorted = np.argsort(sim_array[i])
        sorted = np.flip(sorted)
        for _, j in enumerate(sorted):
            output_file.write(d_list[j]+' ')
        output_file.write('\n')
timestamp()