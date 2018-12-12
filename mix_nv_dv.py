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

eval_set_num = '02'
# use_tfidf = sys.argv[2]
use_tfidf = 1
if use_tfidf == 0:
    nv = KV.load_word2vec_format('nv30k_'+eval_set_num)
else:
    nv = KV.load_word2vec_format('nv30k_tfidf_'+eval_set_num)
dv = doc2vec.Doc2Vec.load('doc2vec_30k_'+eval_set_num+'.model').docvecs

common_idx = np.load('cm_idx_'+eval_set_num+'.npy')
print('updating embeddings...');
alpha = float(sys.argv[1])

# print(nv[t_i['North_Wales']])
for key in tqdm(common_idx):
    nv[key] = alpha*dv[key] + (1-alpha)*nv[t_i[key]]

# print(nv['North_Wales'])
if use_tfidf == 0:
    outfile = 'embeddings/updated_embedding_alpha_'
else:
    outfile = 'embeddings/updated_embedding_tfidf_alpha_'
nv.save_word2vec_format(outfile + str(alpha)+'_'+eval_set_num, binary=True)







#
