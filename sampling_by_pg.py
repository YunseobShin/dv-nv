import networkx as nx
import numpy as np
import json
from time_check import tic
from time_check import toc
import random
import sys

def take_second(e):
    return e[1]

eval_set_num = sys.argv[1]

f = open('index_title.json').read()
i_t = json.loads(f)
f = open('title_index.json').read()
t_i = json.loads(f)

graph_path='edges.txt'
output_file='sample_nodes_77k'+eval_set_num

g = nx.read_edgelist(graph_path, create_using=nx.DiGraph())

print('number of nodes:', g.number_of_nodes())
sample_size = int(input('sample size: '))
sample_size = 77000
print('calculating PageRank...')
tic()
PR = nx.pagerank(g, alpha=0.9)
pr_sorted = sorted([[k,v] for k,v in PR.items()], reverse=True, key=take_second)
pr_sorted = [int(x) for x in np.array(pr_sorted)[:,0]]
np.save('pageranks', pr_sorted)
toc()
tic()
print('sorting by PageRank...')
sample = pr_sorted[:120000]
sample = random.sample(sample, sample_size)
np.save(output_file, sample)
toc()
