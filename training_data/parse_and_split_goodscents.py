import pandas as pd
import spacy
import re
import tqdm
import os
from official.nlp import bert
import official.nlp.bert.tokenization

WRITE_DIRECTORY = "../text"

CSV_FILE = WRITE_DIRECTORY + '/gs_data.csv'
WRITE_FILE = WRITE_DIRECTORY + '/gs_segmented.txt'

df = pd.read_csv(CSV_FILE)

gs_folder_bert = "gs://cloud-tpu-checkpoints/bert/keras_bert/uncased_L-12_H-768_A-12"
tokenizer = bert.tokenization.FullTokenizer(
	vocab_file=os.path.join(gs_folder_bert, "vocab.txt"),
  do_lower_case=True)

def parse_description(data):
	data = str(data)
	# Remove the sentence comment separator and use a period instead.
	data = data.replace("\t\t", ". ")
	# Remove any double periods
	data = data.replace("..", ".")
	# Remove the citations to William Luebke
	data = re.sub(r'luebke, william.*?\(.*?\)\.?', '', str(data))	
	# Remove the citations to Gerard Mosciano
	data = re.sub(r'mosciano, gerard.*?\(.*?\)\.?', '', str(data))	

	sents = []
	doc = nlp(data)
	for sent in doc.sents:
		sents.append(str(sent).strip())

	return sents

nlp = spacy.load("en_core_web_sm")

print("Parsing descriptions into sentences")
comments = [parse_description(data) for data in tqdm.tqdm(df['odor_data'])]

print("Writing to outfile")
with open(WRITE_FILE, 'wt') as outpath:
    for line in tqdm.tqdm(comments):
    	for sent in line:
    		outpath.write(sent + "\n")

    	outpath.write("\n")
