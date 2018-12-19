import json
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors as KV
from tqdm import tqdm
from gensim.models import doc2vec
from gensim.models import Word2Vec
import sys

with open('title_index.json') as f:
    t_i = json.loads(f.read())

with open('index_title.json') as f:
    i_t = json.loads(f.read())

eval_set_num = '01'

nv = KV.load_word2vec_format('nv77k.emb')
dv = doc2vec.Doc2Vec.load('doc2vec_77k_'+eval_set_num+'.model').docvecs

print('extracting common indices...')
nv_keys = list(nv.vocab.keys())
common_idx = [x for x in tqdm(dv.doctags.keys()) if x in nv_keys]
print(len(common_idx))
np.save('cm_idx_'+eval_set_num, common_idx)
