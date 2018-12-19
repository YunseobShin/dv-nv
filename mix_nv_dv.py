import json
import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from gensim.models import KeyedVectors as KV
from tqdm import tqdm
from gensim.models import doc2vec
from gensim.models import Word2Vec
import sys

def mix_nv_dv(eval_set_num, alpha, use_tfidf, t_i, i_t):
    if use_tfidf == 0:
        nv = KV.load_word2vec_format('nv77k.emb')
        print('updating embeddings...');
    else:
        nv = KV.load_word2vec_format('nv77k_tfidf_'+eval_set_num)
        print('updating embeddings with TF-IDF...');
    dv = doc2vec.Doc2Vec.load('doc2vec_77k_'+eval_set_num+'.model').docvecs
    common_idx = np.load('cm_idx_'+eval_set_num+'.npy')
    new_embedding = KV(vector_size = nv.vector_size)
    for key in tqdm(common_idx):
        if key not in dv.doctags.keys():
            continue
        new_embedding[key] = alpha*dv[key] + (1-alpha)*nv[t_i[key]]

    # if use_tfidf == 0:
    #     outfile = 'embeddings/updated_embedding_alpha_'
    # else:
    #     outfile = 'embeddings/updated_embedding_tfidf_alpha_'
    # new_embedding.save_word2vec_format(outfile + str(alpha)+'_'+eval_set_num, binary=True)
    return new_embedding


def lp_nodewise(eval_set_num, alpha, use_tfidf, embeddings, t_i, i_t):
    fname='sample_edges_77k_'+eval_set_num+'.txt'
    g = nx.read_edgelist(fname, create_using=nx.Graph())
    accs = []

    common_idx = np.load('cm_idx_'+eval_set_num+'.npy')
    for node in tqdm(common_idx):
        # print(node)
        neis = [i_t[x] for x in list(g.neighbors(t_i[node]))]
        # print(neis)
        rec = np.array(embeddings.most_similar(node, topn=len(neis)))[:,0]

        # print(rec)
        hits = len(set(neis) & set(rec))
        accs.append(hits/len(neis))

    accuracy = np.mean(accs)
    if use_tfidf == 0:
        print('Accuracy: (alpha='+str(alpha)+'): '+str(accuracy))
    if use_tfidf == 1:
        print('(tfidf)Accuracy: (alpha='+str(alpha)+'): '+str(accuracy))


def main():
    alpha = float(sys.argv[1])
    use_tfidf = 0
    eval_set_num = '01'
    with open('title_index.json') as f:
        t_i = json.loads(f.read())
    with open('index_title.json') as f:
        i_t = json.loads(f.read())
    new_embedding = mix_nv_dv(eval_set_num, alpha, use_tfidf, t_i, i_t)
    lp_nodewise(eval_set_num, alpha, use_tfidf, new_embedding, t_i, i_t)

if __name__ == '__main__':
    main()


#
