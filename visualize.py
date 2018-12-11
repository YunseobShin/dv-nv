import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
import sys
from time_check import tic
from time_check import toc
alpha = sys.argv[1]

dv = KeyedVectors.load('embeddings/updated_embedding_alpha_'+str(alpha)).vectors

model = TSNE(learning_rate=100)
tic()
transformed = model.fit_transform(dv)
toc()
labels = [0]
xs = transformed[:,0]
ys = transformed[:,1]
plt.scatter(xs,ys,c=labels)

plt.show()
