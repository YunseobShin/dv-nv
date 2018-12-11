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
nv = KV.load_word2vec_format('nv30k_'+eval_set_num)
dv = doc2vec.Doc2Vec.load('doc2vec_30k_'+eval_set_num+'.model').docvecs

#####################################################################
print('extracting common indices...')
nv_keys = [i_t[x] for x in list(nv.vocab.keys())]
common_idx = [x for x in tqdm(dv.doctags.keys()) if x in nv_keys]
np.save('cm_idx_'+eval_set_num, common_idx)
print(len(common_idx))
exit()
#####################################################################
common_idx = np.load('cm_idx.npy')
print('updating embeddings...');
print(type(dv))
alpha = float(sys.argv[1])

# print(nv[t_i['North_Wales']])
for key in tqdm(common_idx):
    nv[key] = alpha*dv[key] + (1-alpha)*nv[t_i[key]]

# print(nv['North_Wales'])

nv.save_word2vec_format('embeddings/updated_embedding_alpha_' + str(alpha)+'_'+eval_set_num, binary=True)







#
