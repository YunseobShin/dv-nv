import json

f = open('wiki_dic.json').read()
wiki_dic = json.loads(f)
f = open('title_index.json').read()
t_i = json.loads(f)

p = Popen(['./refine.pl'], stdin=PIPE, stdout=PIPE)
for key in tqdm(wiki_dic):
    page = wiki_dic[key]
    with open('./wikis/'+key+'.txt', 'w') as f:
        f.write(page)
