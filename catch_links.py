import json
import re
from tqdm import tqdm

f = open('wiki_dic.json').read()
wiki_dic = json.loads(f)
f = open('title_index.json').read()
t_i = json.loads(f)

fname = 'edges.txt'
start = open('starts.txt', 'w')
des = open('targets.txt', 'w')

link_file = open(fname, 'w')
link_pattern = re.compile('\[\[(.*?)\]\]')

for title in tqdm(wiki_dic):
    if title in t_i:
        s=t_i[title]
    else:
        break
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
    for link in links:
        link = link.replace(' ', '_')
        if '|' in link:
            link = link.split('|')[0]
        if '#' in link:
            link = link.split('#')[0]
        if len(link) < 1:
            continue
        if link in t_i:
            t=t_i[link]
        else:
            continue
        # start.write(s+'\n')
        # des.write(t+'\n')
        link_file.write(s + ' ' + t + '\n')

link_file.close()
start.close()
des.close()
