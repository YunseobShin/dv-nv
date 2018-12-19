import re
import json
from tqdm import tqdm
import sys
# <redirect title="Télécoms Sans Frontières" />

def file2dic(fname):
    wiki = open(fname, 'r')
    text = wiki.read()
    pages = text.split('</page>')
    wiki_dic = {}
    i_t = {}
    t_i = {}
    for page in tqdm(pages):
        title = re.findall("<title>(.*?)</title>", page)
        id = re.findall("<id>(.*?)</id>", page)
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
        wiki_dic[title] = page
        i_t[id] = title
        t_i[title] = id

    with open('wiki_dic.json', 'w') as w:
        json.dump(wiki_dic, w)
    with open('index_title.json', 'w') as w:
        json.dump(i_t, w)
    with open('title_index.json', 'w') as w:
        json.dump(t_i, w)

if __name__ == '__main__':
    fname = sys.argv[1]
    file2dic(fname)
