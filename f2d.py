import re
import json
from tqdm import tqdm
import sys
# <redirect title="Télécoms Sans Frontières" />

def file2dic(fname):
    wiki = open(fname, 'r')
    text = wiki.read()
    pages = text.split('</doc>')
    wiki_dic = {}
    i_t = {}
    t_i = {}
    for page in tqdm(pages):
        title = re.findall('title=\"(.*?)\"', doc)
        id = re.findall('id=\"(.*?)\"', doc)
        if len(title) > 0:
            title = title[0]
        else:
            continue
        if len(id) > 0:
            id = id[0]
        else:
            continue
        exclude = ['Wikipedia:', 'Template:', 'Category:', '.png', '.jpg', '.bmp', '.gif', '.JGP', '.GIF', '.PNG', '.BMP', '.SWF', '.swf','File:', 'Special:', 'Image:', 'User:', 'User talk:']
        if any(s in title for s in exclude):
            continue
        title = title.replace(' ', '_')
        title = title.replace('/', '-')
        i_t[id] = title
        t_i[title] = id
    with open('index_title.json', 'w') as w:
        json.dump(i_t, w)
    with open('title_index.json', 'w') as w:
        json.dump(t_i, w)

if __name__ == '__main__':
    fname = sys.argv[1]
    file2dic(fname)
