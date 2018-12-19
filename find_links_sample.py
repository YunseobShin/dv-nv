import numpy as np
import sys
from tqdm import tqdm
from time_check import tic
from time_check import toc
import multiprocessing
from multiprocessing import Pool

def write_sample(E):
    eval_set_num = sys.argv[1]
    output = open('sample_edges_77k_'+eval_set_num+'.txt', 'a')
    sample_nodes = np.load('sample_nodes_77k'+eval_set_num+'.npy')
    s=int(E[0])
    t=int(E[1])
    # print(s, t)
    if s in sample_nodes and t in sample_nodes:
        output.write(str(s)+' '+str(t)+'\n')
    output.close()


with open('edges.txt', 'r') as f:
    edges = f.readlines()

edges = [x.split(' ') for x in tqdm(edges)]
eval_set_num = sys.argv[1]
# print(edges[:20])
pool_size = 18
output = open('sample_edges_77k_'+eval_set_num+'.txt', 'w')
output.close()
pool = Pool(processes = pool_size)
tic()
pool.map(write_sample, edges)
toc()
