# coding: utf-8
import networkx as nx
from node2vec import Node2Vec
import numpy as np
import time
import sys

def train(input_file, output_file):
    if len(sys.argv) < 2:
        print('insufficient arguments')
        exit()
    else:
        print('input file:', sys.argv[1], '\noutput file:', sys.argv[2])
    # Create a graph
    print('reading edges...')
    graph = nx.read_edgelist(input_file, create_using=nx.Graph())

    print('number of nodes: ', nx.number_of_nodes(graph))
    print('number of edges: ', nx.number_of_edges(graph))

    # Precompute probabilities and generate walks
    node2vec = Node2Vec(graph, dimensions=128, walk_length=100, num_walks=20, workers=8, p=0.25, q=1)
    # np.save(output_file + 'walks', node2vec.walks)
    model = node2vec.fit(window=6, min_count=1, batch_words=4)  # Any keywords acceptable by gesim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
    # Look for most similar nodes
    # model.wv.most_similar('23979')  # Output node names are always strings
    # indices = model.wv.index2word
    # dimension = len(model.wv['0'])
    # embeddings = np.zeros((len(indices), dimension))
    # embeddings = model.wv
    # for i in range(len(indices)):
    #     embeddings[int(indices[i])] = model.wv[indices[i]]
    # Save embeddings for later use
    model.wv.save_word2vec_format(output_file)
    # output_file = output_file + '.npy'
    # np.save(output_file, embeddings)

    # Save model for later use
    # model.save('ex_model')

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    train(input_file, output_file)
