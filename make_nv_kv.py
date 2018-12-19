import json
import numpy as np
from gensim.models import KeyedVectors as KV
from tqdm import tqdm
from gensim.models import Word2Vec

with open('index_title.json') as f:
    i_t = json.loads(f.read())
f = open('nv77k', 'r')
num_of_nodes, dim = [int(x) for x in f.readline().split()]
nv = KV(vector_size = dim)
for line in tqdm(f.readlines()):
    splits = line.split()
    nv[i_t[splits[0]]] = np.array([float(x) for x in splits[1:]])

nv.save_word2vec_format('nv77k.emb')
