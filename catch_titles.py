import subprocess
import re
from tqdm import tqdm
import numpy as np

def file2dic(fname):
    wiki = open(fname, 'r')
    text = wiki.read()
    pages = text.split('</page>')
    wiki_dic = {}
    for page in tqdm(pages):
        title = re.findall("<title>(.*?)</title>", page)
        if len(title) > 0:
            title = title[0]
        else:
            continue
        exclude = ['Wikipedia:', 'Template:', 'Category:', '.png', '.jpg', '.bmp', '.gif', '.JGP', '.GIF', '.PNG', '.BMP', '.SWF', '.swf','File:', 'Special:', 'Image:', 'User:', 'User talk:']
        if any(s in title for s in exclude):
            continue
        wiki_dic[title] = page

    np.save('wiki_dic', wiki_dic)

if __name__ == '__main__':
    file2dic('sample.txt')
