import numpy as np
import networkx as nx
import json, sys
from tqdm import tqdm

eval_set_num = '02'
with open('title_index.json') as f:
    t_i = json.loads(f.read())
with open('index_title.json') as f:
    i_t = json.loads(f.read())
alpha = sys.argv[1]
sims = np.load('./sorted/sorted_sims'+str(alpha)+'_'+eval_set_num+'.npy')
fname='sample_edges_30k_'+eval_set_num+'.txt'
g = nx.read_edgelist(fname, create_using=nx.Graph())

hit = 0
k = 0
precisions = []
test_nodes = list(np.load('test_nodes.npy'))
sub = nx.Graph()
for e in list(g.edges()):
    if i_t[e[0]] in test_nodes and i_t[e[1]] in test_nodes:
        sub.add_edge(e[0], e[1])

to_predict = sub.number_of_edges()
print(to_predict)
f = open('precision'+str(alpha), 'w')
# print(len(sims))
for sim in sims[:to_predict]:
    k += 1
    if (t_i[sim[0]], t_i[sim[1]]) in list(sub.edges()):
        hit += 1
        precision = hit / k
        precisions.append(precision)

AP = np.mean(precisions)
print('AP (alpha='+str(alpha)+'): ' + str(AP))














#
