import pandas as pd
import spacy
import re
import tqdm
import os

WRITE_DIRECTORY = "../text"

CSV_FILE = WRITE_DIRECTORY + '/leffingwell.csv'
WRITE_FILE = WRITE_DIRECTORY + '/domain_segmented.txt'

GS_FILE = WRITE_DIRECTORY + '/gs_segmented.txt'

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv(CSV_FILE)


def parse_odor_data(data):
    if data == "See comments":
        return [""]

    # Replace semicolons with periods so that we can sometimes parse a sentence.
    data = data.replace(";", ".")
    # Remove any surrounding spaces
    data = data.strip()

    sents = []
    doc = nlp(data.lower())
    for sent in doc.sents:
        sents.append(str(sent).strip())
    return sents


def parse_description(data):
    if data == "nan":
        return [""]

    # Convert to lower for BERT
    data = data.lower()
    # The data randomly contains some quotation marks, so just remove those.
    data = data.replace("\"", '')
    # Remove any surrounding spaces
    data = data.strip()

    # In the comment and occurence fields, the data is already split by double spaces,
    # so just use that instead of NLP
    return data.split("  ")


print("Parsing 'odor_data' from Leffingwell")
comments = [parse_odor_data(str(data)) for data in tqdm.tqdm(df['odor_data'])]
print("Parsing 'comment' from Leffingwell")
comments2 = [parse_description(str(data)) for data in tqdm.tqdm(df['comment'])]
print("Parsing 'occurrence' from Leffingwell")
comments3 = [
    parse_description(str(data)) for data in tqdm.tqdm(df['occurrence'])
]

with open(WRITE_FILE, 'wt') as outpath:
    print("Writing Goodscents file")
    with open(GS_FILE, "rt") as gs:
        for raw in tqdm.tqdm(gs.readlines()):
            outpath.write(raw)

    print("Writing Leffingwell file")
    for i in tqdm.tqdm(range(len(comments))):
        all_data = comments[i] + comments2[i] + comments3[i]
        for sent in all_data:
            if sent == "" or sent.isspace():
                continue
            outpath.write(sent + "\n")

        outpath.write("\n")
