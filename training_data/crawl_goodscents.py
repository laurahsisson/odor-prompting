from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import re
from IPython.display import display
import cirpy
import urllib.error 

import rdkit
from rdkit.Chem import AllChem as Chem

import requests_cache
import requests

WRITE_DIRECTORY = "../text"

requests_cache.install_cache(cache_name= WRITE_DIRECTORY + '/gs_cache', backend='sqlite', expire_after=180)

ALL_LINKS_FILE = WRITE_DIRECTORY + '/gs_cas_links.txt'

CSV_FILE = WRITE_DIRECTORY + '/gs_data.csv'


print("Parse all URLS")
base_url = 'http://www.thegoodscentscompany.com/allprod-{}.html'
links = []

all_pages = ['a','b','c','d','e','f','g','h','i','jk','l','m','n','o',
             'p','q','r','s','t','u','v','wx','y','z']
for num in tqdm(all_pages):
    url = base_url.format(num)
    resp = requests.get(url)
    html = resp.content
    soup = bs(html,"lxml")
    for a in soup.find_all('a', href=True):
        if a['href'] == "#" and a.get('onclick'):
            alink = str(a["onclick"])
            alink = alink.replace("openMainWindow('","").replace("');return false;","")
            links.append(alink)
links = list(set(links))
print(f'Found {len(links)} links')
with open(ALL_LINKS_FILE, 'w') as afile:
    afile.write('\n'.join(links))


print("Read all links")
info_df = pd.read_csv(ALL_LINKS_FILE, names=['url'])
print(info_df.shape)

def parse_one_page(url):
    info_dict = {'url':url, 'useful':True}
    resp = requests.get(url)
    html = resp.content
    if 'Odor Description:' not in str(html) and 'Taste Description:' not in str(html):
        info_dict['useful'] = False
        return info_dict
    soup = bs(html, "lxml")
    for br in soup.findAll('br'):
        br.replace_with(" " + br.text)
        
    try:
        name = soup.find(attrs={"itemprop":"name"}).get_text()
        info_dict.update({'name':name})
    except:
        pass
        
    odor_descriptions = []
    for match in soup.find_all(string=re.compile("(Odor Description)|(Taste Description)")):
        try:
            parent = match.parent
            text = []
            for span in parent.find_all("span"):
                text.append(span.get_text())
            description = ''.join(text)
            description = description.replace('\n','').replace('\t','').replace('\r','')
            odor_descriptions.append(description.lower())
        except:
            continue
    if len(odor_descriptions) == 0:
        info_dict.update({'useful':False})
    else: 
        odor_data = "\t\t".join(odor_descriptions)
        info_dict.update({'odor_data':odor_data})
    return info_dict
    
print("Print an example page")
url = 'http://www.thegoodscentscompany.com/data/rw1007872.html'
result = parse_one_page(url)
print(result["odor_data"])

print("Write to CSV")
results=[]
for index, row in tqdm(info_df.iterrows(), total=len(info_df)):
    results.append(parse_one_page(row['url']))

results_df = pd.DataFrame(results)
useful_df = results_df.query('useful')
useful_df.to_csv(CSV_FILE,index=False)
