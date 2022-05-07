import spacy
from spacy.symbols import ORTH, LEMMA, POS
import unittest
import matplotlib.pyplot as plt
from collections import Counter
import tqdm
import numpy as np

import string

nlp = spacy.load("en_core_web_trf")

DOMAIN_RAW = "domain_segmented.txt"
DOMAIN_LEMMATIZED = "domain_lemmatized.txt"

BASENOTES_RAW = "basenotes_segmented.txt"
BASENOTES_LEMMATIZED = "basenotes_lemmatized.txt"

COMBINED_FILE = "all_word_chunks.txt"

WORD_SEP = ", "

def clean(text):
    text = text.strip().lower()
    text = text.replace('-', ' ')
    text = "".join(
        [c for c in text if c in string.ascii_lowercase or c == ' '])
    return text

nlp.add_pipe("merge_noun_chunks")

FROM_BASENOTES = 0

def get_spacy_noun_chunks(sentence):
    global FROM_BASENOTES
    doc = nlp(sentence)
    chunks = []
    for t in doc:
        if "NN" in t.tag_ or t.text in all_descriptors:
            chunks.append(t.text)
            if t.text in all_descriptors and "NN" not in t.tag_:
                FROM_BASENOTES += 1
    return chunks


def get_lemmatized_docs(read_file, get_chunks_fn):
    i = 0
    num_lines = sum(1 for line in open(read_file, 'r'))
    all_docs = []

    with open(read_file) as f:
        docs = []
        for line in tqdm.tqdm(f, total=num_lines):
            if line == "\n":
                if docs != []:
                    all_docs.append(docs)
                docs = []
                continue

            line = line.strip()
            chunks = get_chunks_fn(line)
            chunks = [clean(c) for c in chunks]
            for chunk in chunks:
                if chunk != "":
                    docs.append(chunk)

    return all_docs

def write_docs_to_file(all_docs, out_file):
    with open(out_file, 'wt') as outpath:
        for all_docs in tqdm.tqdm(all_docs):
            if not all_docs:
                continue

            for word in all_docs:
                outpath.write(word + WORD_SEP)

            outpath.write("\n")
            outpath.write("\n")


# Drop every word that occurs only once.
# This reduces the unique words from 40k to 10k,
# but only loses around 8% of the total occurences
def remove_singles(all_docs):
    occurs = Counter()

    # Count up every occurence
    for d in all_docs:
        for c in d:
            occurs[c] += 1

    # Only include those that occur more than once
    return [[c for c in d if occurs[c] > 1] for d in all_docs]


print("Reading from Basenotes.")
basenotes_docs = get_lemmatized_docs(BASENOTES_RAW,
                                     lambda sent: sent.split(","))

all_descriptors = set()

print("Updating spaCy descriptors.")
for doc in tqdm.tqdm(basenotes_docs):
    for descriptor in doc:
        nlp.tokenizer.add_special_case(str(descriptor), [{
            ORTH: str(descriptor),
        }])
        all_descriptors.add(descriptor)

print(f"Found a total of {len(all_descriptors)} descriptors.")

domain_docs = get_lemmatized_docs(DOMAIN_RAW, get_spacy_noun_chunks)
print(f"Added an extra {FROM_BASENOTES}.")

write_docs_to_file(basenotes_docs, BASENOTES_LEMMATIZED)
write_docs_to_file(domain_docs, DOMAIN_LEMMATIZED)

all_docs = domain_docs + basenotes_docs
write_docs_to_file(all_docs, COMBINED_FILE)

# Two places to go from here.
# Drop everything that occurs only once
# Take everything that occurs 5 times as a token, then use the tokenization scheme propsoed by Ashish
