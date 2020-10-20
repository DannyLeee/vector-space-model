#%% 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

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
def file_iter():
    for name in q_list:
        with open(query_path+name+'.txt') as f:
            yield f.readline()
    for name in d_list:
        with open(doc_path+name+'.txt') as f:
            yield f.readline()

vectorizer = TfidfVectorizer(sublinear_tf=False, stop_words=None, token_pattern="(?u)\\b\\w+\\b", smooth_idf=True, norm='l2')
tfidf = vectorizer.fit_transform(file_iter())    # input string of list
tfidf = tfidf.toarray()

#%%
df_q = pd.DataFrame(tfidf[ : len(q_list)],columns=vectorizer.get_feature_names(), index=q_list)
df_d = pd.DataFrame(tfidf[len(q_list) : ],columns=vectorizer.get_feature_names(), index=d_list)


#%%
sim_array = cosine_similarity(df_q.loc[q_list], df_d.loc[d_list])
sim_array = np.array(sim_array)

#%%
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

#%%=========================================
t = cosine_similarity(df_q.loc[['301','302','303']], df_d.loc[['FBIS4-68801', 'FBIS3-23986']])
t[0][0]

#%%
df_d.loc['FBIS4-68801']

#%%
cosine_similarity([df_q.loc['301']], [df_d.loc['FBIS4-68801']])

#%%
d1 = 'a b d e d f a f e fa d s a b n'
d2 = 'a z a f e fa h'
d3 = 'a z a f e fa h'
vectorizer = TfidfVectorizer(sublinear_tf=False, stop_words=None, token_pattern="(?u)\\b\\w+\\b", smooth_idf=True, norm='l2')
tfidf = vectorizer.fit_transform([d1,d2,d3])
df_tfidf = pd.DataFrame(tfidf.toarray(),columns=vectorizer.get_feature_names(), index=['d1', 'd2', 'd3'])
print("TFIDF")
df_tfidf


# %%
# Vectors
vec_a = [1, 2, 3, 4, 5]
vec_b = [1, 3, 5, 7, 9]
vec_c = [1, 2, 3, 7, 5]
vec_d = [1, 3, 5, 7, 15]

# Dot and norm
dot = sum(a*b for a, b in zip(vec_a, vec_b))
norm_a = sum(a*a for a in vec_a) ** 0.5
norm_b = sum(b*b for b in vec_b) ** 0.5

# Cosine similarity
cos_sim = dot / (norm_a*norm_b)

# Results
print('My version:', cos_sim)
print('Scikit-Learn:', cosine_similarity([vec_a], [vec_b, vec_c, vec_d]))
