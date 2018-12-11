import json
import re
from tqdm import tqdm

def clean_str(string):
    string = re.sub('<.*>', '', string)
    string = re.sub('&amp', '&', string)
    string = re.sub('&it', '<', string)
    string = re.sub('&gt', '>', string)
    string = re.sub('<ref>[^<]*<\/ref>', '', string)
    string = re.sub('<[^>]*>', '', string)
    string = re.sub('\[http:[^] ]*', '[', string)
    string = re.sub('\|thumb', '', string)
    string = re.sub('\|left', '', string)
    string = re.sub('\|right', '', string)
    string = re.sub('\|d+px', '', string)
    string = re.sub('\[\[image:[^\[\]]*\|/', '', string)
    string = re.sub('\[\[category:([^|\]]*)[^]]*\]\]', '', string)
    string = re.sub('\[\[[a-z\-]*:[^\]]*\]\]', '', string)
    string = re.sub('\[\[[^\|\]]*\|', '', string)
    string = re.sub('{{[^}]*}}', '', string)
    string = re.sub('{[^}]*}', '', string)
    string = re.sub('\[//', '', string)
    string = re.sub('\]//', '', string)
    string = re.sub('&[^;]*;', ' ', string)

    string = string.strip().lower() + '\n'
    string = string.replace('1', ' one ')
    string = string.replace('2', ' two ')
    string = string.replace('3', ' three ')
    string = string.replace('4', ' four ')
    string = string.replace('5', ' five ')
    string = string.replace('6', ' six ')
    string = string.replace('7', ' seven ')
    string = string.replace('8', ' eight ')
    string = string.replace('9', ' nine ')
    string = string.replace('0', ' zero ')

    if string[0] == '\n':
        return ''
    else:
        return string

f = open('wiki_dic.json').read()
wiki_dic = json.loads(f)
f = open('title_index.json').read()
t_i = json.loads(f)

for key in tqdm(wiki_dic):
    page = wiki_dic[key]
    page = clean_str(page)
    with open('./wikis/'+key+'.txt', 'w') as f:
        f.write(page.encode('utf-8'))
