import numpy as np
import sys, json
import networkx as nx
from time_check import tic
from time_check import toc
from tqdm import tqdm
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
import random

def take_thrid(e):
    return e[2]

def take_second(e):
    return e[1]

def sample_tests(fname, sample_size, e):
    g = nx.read_edgelist(fname, create_using=nx.DiGraph())
    with open('index_title.json') as f:
        i_t = json.loads(f.read())
    PR = nx.pagerank(g, alpha=0.9)
    pr_sorted = sorted([[k,v] for k,v in PR.items()], reverse=True, key=take_second)
    pr_sorted = [int(x) for x in np.array(pr_sorted)[:,0]]
    sample = pr_sorted
    sample = [i_t[str(x)] for x in sample]
    keys = np.load('cm_idx_'+e+'.npy')
    sample = [x for x in sample if x in keys]
    sample = random.sample(sample, sample_size)
    return sample

def get_pairs_by_cosine(test_nodes, dv):
    sims = []
    for k in tqdm(test_nodes):
        for j in test_nodes:
            if k == j:
                continue
            sims.append([k, j, cosine(dv[k], dv[j])])
        test_nodes = [x for x in test_nodes if x != k]
    sims = np.array(sims)
    sims = np.array(sorted(sims, key=take_thrid))[:,:2]
    return sims

def get_pairs_by_Hadamard(test_nodes, dv):
    with open('index_title.json') as f:
        i_t = json.loads(f.read())
    with open('title_index.json') as f:
        t_i = json.loads(f.read())
    g = nx.read_edgelist(fname, create_using=nx.Graph())
    sub = nx.Graph()
    for e in list(g.edges()):
        if i_t[e[0]] in test_nodes and i_t[e[1]] in test_nodes:
            sub.add_edge(e[0], e[1])
    edge_features = []
    labels = []
    for k in tqdm(test_nodes):
        for j in test_nodes:
            if k == j:
                continue
            edge_features.append(dv[k]*dv[j])
            if (t_i[k], t_i[j]) in list(sub.edges()):
                labels.append(1)
            else:
                labels.append(0)
        test_nodes = [x for x in test_nodes if x != k]

    return edge_features, labels

eval_set_num = '01'
fname='sample_edges_30k_'+eval_set_num+'.txt'
sample_size=10000
test_nodes = sample_tests(fname, sample_size, eval_set_num)
np.save('test_nodes', test_nodes)
exit()
test_nodes = np.load('test_nodes.npy')
print('lenght of test nodes:', len(test_nodes))

alpha = sys.argv[1]
dv = KeyedVectors.load('embeddings/updated_embedding_alpha_' + str(alpha)+'_'+eval_set_num)
sims = get_pairs_by_cosine(test_nodes, dv)
np.save('./sorted/sorted_sims'+str(alpha)+'_'+eval_set_num, sims)
edge_features, labels = get_pairs_by_Hadamard(test_nodes, dv)
np.save('./edges/edge_features'+str(alpha), edge_features)
np.save('./edges/edge_labels'+str(alpha), labels)
