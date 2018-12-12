import json
import re, math
import networkx as nx
from tqdm import tqdm
import collections
import numpy as np

with open('wiki_dic.json', 'r') as f:
    wiki_dic = json.loads(f.read())
with open('title_index.json') as f:
    t_i = json.loads(f.read())
with open('index_title.json') as f:
    i_t = json.loads(f.read())

eval_set_num = '03'
fname ='sample_edges_30k_'+eval_set_num+'.txt'
g = nx.read_edgelist(fname, create_using=nx.DiGraph())
output = 'edges_tfidf_weights_'+eval_set_num+'.txt'
outfile = open(output, 'w')
titles = [i_t[x] for x in list(g.nodes())]
link_pattern = re.compile('\[\[(.*?)\]\]')
idf = g.in_degree()

for title in tqdm(titles):
    page = wiki_dic[title]
    links = link_pattern.findall(page)
    exclude = ['File:', 'Special:', 'Category:', 'Image:', 'User:', 'User talk:', 'Template:']
    to_delete = []
    for link in links:
        if any(s in link for s in exclude):
            to_delete.append(link)
    for link in to_delete:
        if link in links:
            links.remove(link)

    counter=collections.Counter(links)
    s = t_i[title]
    ts = []
    ws = []
    for link in counter:
        link = link.replace(' ', '_')
        if '|' in link:
            link = link.split('|')[0]
        if '#' in link:
            link = link.split('#')[0]
        if len(link) < 1:
            continue
        if link in t_i:
            if link in titles:
                ts.append(t_i[link])
                w = 0.5 + counter[link] / (math.log(idf[t_i[link]]) + 1)
                ws.append(w)

    for i in range(len(ws)):
        ws[i] /= np.sum(ws)
        outfile.write(s + ' ' + ts[i] + ' ' + str(ws[i]) + '\n')

outfile.close()
















#
