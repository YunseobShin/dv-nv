#-*- coding: utf-8 -*-
from gensim.models import doc2vec
import sys, json
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument
import nltk
import pickle
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import multiprocessing
cores = multiprocessing.cpu_count()

#doc2vec parameters
vector_size = 128
window_size = 15
word_min_count = 2
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 1 #0 = dbow; 1 = dmpv
worker_count = cores #number of parallel processes
eval_set_num = '02'
# 'title_contents_##.json'
# 'doc2vec_30k_##'
if len(sys.argv) >= 3:
	inputfile = sys.argv[1]
	modelfile = sys.argv[2]

else:
	inputfile = "./title_contents.json"
	modelfile = "./doc2vec_30k.model"

with open(inputfile, 'r', encoding='UTF8') as f:
    data = json.loads(f.read())
ks = data.keys()
vs = data.values()

sentences = []

for k in tqdm(data):
    sentences.append(TaggedDocument(words=word_tokenize(data[k].lower()), tags=[k]))
with open('sentences_'+eval_set_num+'.pkl', 'wb') as g:
	pickle.dump(sentences, g)

# with open('sentences_'+eval_set_num+'.pkl', 'rb') as g:
# 	sentences = pickle.loads(g.read())

doc_vectorizer = doc2vec.Doc2Vec(min_count=word_min_count, size=vector_size, alpha=0.025, min_alpha=0.025, seed=1234, workers=worker_count)
doc_vectorizer.build_vocab(sentences)
doc_vectorizer.train_words = False
doc_vectorizer.train_lbls = True
doc_vectorizer.train(sentences, epochs=30, total_examples=doc_vectorizer.corpus_count)

doc_vectorizer.save(modelfile)
