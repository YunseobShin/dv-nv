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
    sub = graph.subgraph(max(nx.connected_components(graph), key=len))
    print('number of nodes: ', nx.number_of_nodes(sub))
    print('number of edges: ', nx.number_of_edges(sub))

    # Precompute probabilities and generate walks
    node2vec = Node2Vec(sub, dimensions=256, walk_length=80, num_walks=20, workers=8, p=0.25, q=1)
    # np.save(output_file + 'walks', node2vec.walks)
    model = node2vec.fit(window=15, min_count=1, batch_words=4)  # Any keywords acceptable by gesim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

    model.wv.save_word2vec_format(output_file)

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    train(input_file, output_file)
