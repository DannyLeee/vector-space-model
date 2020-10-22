#%% 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

#%%
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
from datetime import datetime,timezone,timedelta
def timestamp():
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(dt2)

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
timestamp()
df_tf = pd.DataFrame(list_tf)
timestamp()

#%%
#idf
df_idf = pd.DataFrame([np.log(1+(len(df_tf)+1) / (df_tf.count()+1))], index=['idf'])

#%%
df_tf.insert(0, column='txt_id', value=q_list+d_list)
df_tf = df_tf.set_index('txt_id')
df_tf = df_tf.fillna(0)
print('\n')
print(df_tf.shape)
print(df_idf.shape)

#%%
# np_tfidf
timestamp()
np_tf = np.array(df_tf)
np_idf = np.array(df_idf)
# np_tfidf = (1+np.log(np_tf)) * np_idf
np_tfidf = np_tf * np_idf
timestamp()
np_tfidf

#%%
df_tfidf.to_csv('tfidf.csv')

#%%
# sim_array
timestamp()
sim_array = cosine_similarity(np_tfidf[:len(q_list)], np_tf[len(q_list):])
sim_array = np.array(sim_array)
timestamp()
sim_array

#%%###########################
vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words=None, token_pattern="(?u)\\b\\w+\\b", smooth_idf=True, norm='l2')
tfidf = vectorizer.fit_transform(file_iter())    # input string of list

#%%
np_tfidf[51]

#%%
tfidf[51].toarray()

#%%
sim_array = cosine_similarity(tfidf[:len(q_list)], tfidf[len(q_list):])
sim_array = np.array(sim_array)
sim_array

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
        # break