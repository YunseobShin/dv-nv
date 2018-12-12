import time
from time_check import tic
from time_check import toc
import json
import numpy as np
from numpy import array
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors as NN
from scipy.spatial.distance import euclidean as dis
import sys
import multiprocessing
from multiprocessing import Pool
from functools import partial
from contextlib import contextmanager

@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def make_graph(filename):
    with open(filename, 'r') as data:
        edges = []
        for line in data:
            t = tuple(line.split(','))
            edges.append(t)
    return edges

def get_degrees(dges, is_directed, length):
    for i in range(length):
        edges[i] = [int(x) for x in edges[i]]
    edges = np.array(edges)
    res = {}
    # directed: 'node_id': [OUT_deg, in_deg]
    if is_directed == 1:
        for edge in edges:
            s = str(edge[0])
            d = str(edge[1])

            if s in res:
                res[s][0] += 1
            else:
                res[s] = [1, 0]
            if d in res:
                res[d][1] += 1
            else:
                res[d] = [0, 1]

    # undirected
    elif is_directed == 2:
        sum = 0
        for i in range(1, length + 1):
            res[str(i)] = len(np.where(edges[:,0] == i)[0])
            res[str(i)] += len(np.where(edges[:,1] == i)[0])
            res[str(i)] /= 2
            sum += res[str(i)]
            if i % 50000 == 0 :
                print(str(i)+'/'+str(length))
                toc()
                tic()
        print(sum)
    return res

def get_treshold(vecs):
    mu = float(input('Enter the threshold hyperparameter mu: '))
    sample = vecs[np.random.choice(vecs.shape[0], 8800, replace=False), :]
    distances = []
    len = sample.shape[1]
    for i in range(len):
        for j in range(len):
            distances.append(dis(sample[i], sample[j]))
    res = np.mean(array(distances))
    print('The average distance of sample: ', res)
    res /= mu
    return res

def split_vecs(vecs, pool_size):
    length = vecs.shape[0]
    size = length // pool_size
    margin = length % pool_size
    s = 0
    t = size
    res = []
    for i in range(pool_size-1):
        res.append(vecs[s:t])
        s += size
        t += size
        res.append(vecs[s:length])
        return res

# def check_link_by_degree(v, degrees):

def check_link_deg(vecs, v, v_id, degrees):
    deg = degrees[str(v_id+1)]
    # print(deg)
    nh = NN(n_neighbors=deg, algorithm='ball_tree')
    nh.fit(vecs)
    v = [list(v)]
    ng = nh.kneighbors(v, return_distance=False)[0]
    g = open('new_graph_deg.txt', 'a')
    for index in ng:
        g.write(str(v_id+1)+' '+ str(index+1)+'\n')
    g.close()

def check_link_by_distance(v, s, threshold):
    res = dis(v, s) < threshold
    if res.all():
        return True
    else:
        return False

def parallel_check_link_dis(vecs, v, v_id, threshold):
    g = open('new_graph.txt', 'a')
    nhr = NN(radius=threshold)
    nhr.fit(vecs)
    v = [list(v)]
    rng = nhr.radius_neighbors(v, return_distance=False)[0]
    for index in rng:
        if v_id == index:
            continue
        # assume the graph is undirected.
        g.write(str(v_id+1)+' '+ str(index+1)+'\n')
    g.close()

def parallel_check_link_deg(V, vecs, degrees, fname):
    # print('3ㅅ3')
    v = array(V[0])
    v_id = V[1][0]
    # print('vid: ', v_id+1)
    deg = degrees[str(v_id+1)]
    # print(deg)
    nh = NN(n_neighbors=deg, algorithm='ball_tree')
    nh.fit(vecs)
    v = [list(v)]
    ng = nh.kneighbors(v, return_distance=False)[0]
    g = open(fname, 'a')
    for index in ng:
        g.write(str(v_id+1)+' '+ str(index+1)+'\n')
    g.close()

def vec_to_node(vecs, length, pool_size, cri, fname):
    print('vec_to_node')
    if cri == 1:
        threshold = get_treshold(vecs)
        print('the distance threshold:', threshold)
        print('converting vectors back into a graph by distance threshold...')
        chunks = split_vecs(vecs, pool_size)
        # print(chunks)
        # print(array(chunks).shape)
    else:
        print('converting vectors back into a graph by original degrees...')
        d = int(input('recon number: '))
        j = open('degrees/'+'recon_degree'+str(d)+'.json', 'r')
        degrees = json.load(j)
        # degrees = get_degrees(make_graph('BlogCatalog/data/edges.csv'), 2)
        ids = np.arange(length)
        V = []
        for i in range(length):
            V.append([ list(vecs[i]), [ids[i]] ])
        # print(V[0:50])

    tic()

    if cri == 1:
        for i in range(length):
            with poolcontext(processes=pool_size) as pool:
                pool.map(partial(parallel_check_link_dis, v = vecs[i], v_id = i, threshold = threshold), chunks)
            if i % 20 == 0 : # for mesauring computational time tests
            # if i % 5000 == 0 : # for actual executions
                print(str(i)+'/'+str(length))
                toc()
                tic()
    else:
        tic()
        with poolcontext(processes=pool_size) as pool:
            pool.map(partial(parallel_check_link_deg, vecs = vecs, degrees = degrees, fname = fname), V)
        toc()


if __name__ == '__main__':
    pool_size = int(input('Enter the pool size: '))
    while True:
        cri = int(input('1=>distance, 2=>degree: '))
        if cri in [1,2]:
            break
    outfile = sys.argv[2]
    g = open(outfile, 'w')

    print('You are using', pool_size, 'processesors in', multiprocessing.cpu_count(), 'processesors.')
    tic()
    infile = sys.argv[1]
    with open(infile, 'r') as em:
        vec = json.load(em)
        vecs = array(list(vec.values()))
        # vec_to_node(vecs, 4)
    length = vecs.shape[0]
    vec_to_node(vecs, length, pool_size, cri, outfile)

    toc()
    g.close()







# def shilouette(vecs, k):
    #sil_scores=[]
    # for k in range(5, 21):
    #     print('clustering...')
    #     tic()
    #     kmeans = KMeans(n_clusters = k, random_state=0).fit(vecs)
    #     labels = kmeans.labels_
    #     toc()
    #     print('Calculating Silhouette score...')
    #     sil_score = silhouette_score(vecs, labels)
    #
    #     print(sil_score)
    #     sil_scores.append(sil_score)
    # with open('sil_scores.txt', 'w') as sil:
    #     sil.write(sil_scores)





###################
