#%% 
from sklearn.feature_extraction.text import TfidfVectorizer
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

#%%
timestamp()
doc_path = "./data/docs/"
query_path = "./data/queries/"

#%%
d_list = []
with open('./data/doc_list.txt', 'r') as d_list_file:
    for line in d_list_file:
        line = line.replace("\n", "")
        d_list += [line]

#%%
q_list = []
with open('./data/query_list.txt', 'r') as q_list_file:
    for line in q_list_file:
        line = line.replace("\n", "")
        q_list += [line]

#%%
def file_iter():
    for name in q_list:
        with open(query_path+name+'.txt') as f:
            yield f.readline()
    for name in d_list:
        with open(doc_path+name+'.txt') as f:
            yield f.readline()

#%%
doc_list = []
list_tf = []
for txt in tqdm(file_iter()):
    list_tf += [Counter(txt.split())]

#%%
# list of dict to dataframe
df_tf = pd.DataFrame(list_tf)
df_q_tf = df_tf[:len(q_list)]
df_d_tf = df_tf[len(q_list):]
del df_tf

#%%
#idf
df_idf = pd.DataFrame([np.log(1+(len(df_d_tf)+1) / (df_d_tf.count()+1))], index=['idf'])

#%%
df_q_tf.insert(0, column='txt_id', value=q_list)
df_d_tf.insert(0, column='txt_id', value=d_list)
df_q_tf = df_q_tf.set_index('txt_id')
df_q_tf = df_q_tf.fillna(0)
df_d_tf = df_d_tf.set_index('txt_id')
df_d_tf = df_d_tf.fillna(0)
print('\n')
print(df_q_tf.shape)
print(df_d_tf.shape)
print(df_idf.shape)

#%%
# np_tfidf
np_q_tf = np.array(df_q_tf)
np_d_tf = np.array(df_d_tf)
np_idf = np.array(df_idf)
np_q_tfidf = (1 + np.ma.log2(np_q_tf)) * np_idf
np_d_tfidf = (1 + np.ma.log2(np_d_tf)) * np_idf
del df_idf, np_q_tf, np_d_tf

#%%
np_q_tfidf = np_q_tfidf.filled(0)
np_d_tfidf = np_d_tfidf.filled(0)

#%%
# sim_array
sim_array = cosine_similarity(np_q_tfidf, np_d_tfidf)
sim_array = np.array(sim_array)

#%%!!!!!!!!!!!!!!!!!!!!!!!!!
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