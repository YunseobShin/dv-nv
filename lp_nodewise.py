import numpy as np
import networkx as nx
import json, sys
from tqdm import tqdm
from gensim.models import KeyedVectors

alpha = sys.argv[1]
eval_set_num = '01'
with open('title_index.json') as f:
    t_i = json.loads(f.read())
with open('index_title.json') as f:
    i_t = json.loads(f.read())

fname='sample_edges_30k_'+eval_set_num+'.txt'
g = nx.read_edgelist(fname, create_using=nx.Graph())

test_nodes = list(np.load('test_nodes.npy'))
embeddings = KeyedVectors.load_word2vec_format('embeddings/updated_embedding_alpha_' + str(alpha)+'_'+eval_set_num, binary=True)
accs = []
# print(embeddings['North_Wales'])
common_idx = np.load('cm_idx_'+eval_set_num+'.npy')
for node in tqdm(common_idx):
    # print(node)
    neis = [i_t[x] for x in list(g.neighbors(t_i[node]))]
    # print(neis)
    rec = np.array(embeddings.most_similar(node, topn=len(neis)))[:,0]
    # print(rec)
    hits = len(set(neis) & set(rec))
    accs.append(hits/len(neis))

# for node in test_nodes:
#     # print(node)
#     neis = [i_t[x] for x in list(g.neighbors(t_i[node]))]
#     # print(neis)
#     rec = np.array(embeddings.most_similar(node, topn=len(neis)))[:,0]
#     # print(rec)
#     hits = len(set(neis) & set(rec))
#     accs.append(hits/len(neis))

accuracy = np.mean(accs)
print('Accuracy: (alpha='+str(alpha)+'): '+str(accuracy))
