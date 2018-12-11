import networkx as nx
import numpy as np
import json, sys
from tqdm import tqdm
import random
import multiprocessing
from multiprocessing import Pool

def write_sample(E):
    eval_set_num = sys.argv[1]
    output = open('sample_edges_30k_'+eval_set_num+'.txt', 'a')
    sample_nodes = np.load('sample_nodes_30k'+eval_set_num+'.npy')
    s=int(E[0])
    t=E[1]
    t = int(t.replace('\n', ''))
    # print(s, t)
    if s in sample_nodes and t in sample_nodes:
        output.write(str(s)+' '+str(t)+'\n')
    output.close()


with open('edges.txt', 'r') as f:
    edges = f.readlines()
edges = [x.split(' ') for x in tqdm(edges)]
eval_set_num = sys.argv[1]
# print(edges[:20])
pool_size = 24
output = open('sample_edges_30k_'+eval_set_num+'.txt', 'w')
output.close()
pool = Pool(processes = pool_size)
pool.map(write_sample, edges)
