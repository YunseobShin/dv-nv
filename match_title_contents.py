import json, re, sys
from tqdm import tqdm
import numpy as np

eval_set_num = sys.argv[1]
sample_nodes = np.load('sample_nodes_77k'+eval_set_num+'.npy')
sample_nodes = [int(x) for x in sample_nodes]
fn = open('wiki', 'r')
wiki = fn.read()
docs = wiki.split('</doc>')
title_contents={}
output=open('wiki_texts', 'w')

for doc in tqdm(docs):
    if len(re.findall('title=\"(.*?)\"', doc)) > 0:
        title = re.findall('title=\"(.*?)\"', doc)[0]
    else:
        continue
    if len(re.findall('id=\"(.*?)\"', doc)) > 0:
        id = re.findall('id=\"(.*?)\"', doc)[0]
    else:
        continue

    title = title.replace(' ', '_')
    title = title.replace('/', '-')

    if int(id) in sample_nodes:
        # print(title)
        contents = re.sub('<doc(.*?)>', '', doc)
        contents = contents.replace('\n', ' ')
        contents = re.sub(r'[\W_]+', ' ', contents)
        contents = " ".join(contents.split())
        contents = contents.lower()
        contents = contents.replace(' a ', ' ')
        title_contents[title] = contents

print(len(title_contents))
with open('title_contents_77k_'+eval_set_num+'.json', 'w') as f:
    json.dump(title_contents, f)
fn.close()
output.close()
