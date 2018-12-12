import numpy as np
import networkx as nx
import json, sys
from tqdm import tqdm
from gensim.models import KeyedVectors

alpha = sys.argv[1]
eval_set_num = '02'
use_tfidf = 1
with open('title_index.json') as f:
    t_i = json.loads(f.read())
with open('index_title.json') as f:
    i_t = json.loads(f.read())

fname='sample_edges_30k_'+eval_set_num+'.txt'
g = nx.read_edgelist(fname, create_using=nx.Graph())
if use_tfidf == 0:
    infile = 'embeddings/updated_embedding_alpha_'
else:
    infile = 'embeddings/updated_embedding_tfidf_alpha_'
embeddings = KeyedVectors.load_word2vec_format(infile + str(alpha)+'_'+eval_set_num, binary=True)
accs = []

common_idx = np.load('cm_idx_'+eval_set_num+'.npy')
for node in tqdm(common_idx):
    # print(node)
    neis = [i_t[x] for x in list(g.neighbors(t_i[node]))]
    # print(neis)
    rec = np.array(embeddings.most_similar(node, topn=len(neis)))[:,0]
    # print(rec)
    hits = len(set(neis) & set(rec))
    accs.append(hits/len(neis))

accuracy = np.mean(accs)
if use_tfidf == 0:
    print('Accuracy: (alpha='+str(alpha)+'): '+str(accuracy))
if use_tfidf == 1:
    print('(tfidf)Accuracy: (alpha='+str(alpha)+'): '+str(accuracy))
