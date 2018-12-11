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

# 'title_contents_##.json'
# 'doc2vec_30k_##.model'
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
with open('sentences.pkl', 'wb') as g:
	pickle.dump(sentences, g)

# with open('sentences.pkl', 'rb') as g:
# 	sentences = pickle.loads(g.read())

# sentences = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)])
#                 for i, _d in (data.keys(), data.values())]

word2vec_file = modelfile + ".word2vec_format"

# sentences=doc2vec.TaggedLineDocument(inputfile)
#build voca

doc_vectorizer = doc2vec.Doc2Vec(min_count=word_min_count, size=vector_size, alpha=0.025, min_alpha=0.025, seed=1234, workers=worker_count)
doc_vectorizer.build_vocab(sentences)
doc_vectorizer.train_words = False
doc_vectorizer.train_lbls = True
doc_vectorizer.train(sentences, epochs=20, total_examples=doc_vectorizer.corpus_count)

# Train document vectors!
# for epoch in range(10):
# 	doc_vectorizer.train(sentences)
# 	doc_vectorizer.alpha -= 0.002 # decrease the learning rate
# 	doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay

# To save
doc_vectorizer.save(modelfile)
doc_vectorizer.save_word2vec_format(word2vec_file, binary=False)
