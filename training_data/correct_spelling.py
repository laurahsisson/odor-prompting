import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from fuzzywuzzy import fuzz, process
import tqdm
from collections import Counter
import numpy as np
import pandas as pd

IMPORT_FILE = "data/domain_lemmatized.txt"
CORRECTED_FILE = "data/domain_lemmatized.txt"

WORD_SEP = ", "

MODEL_NAME = "distilroberta-base"
TEST_FILE = "test_word_list.txt"

MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

remove_words = set([
    "a", "an", "the", "essence", "oil", "like", "undertone", "note", "juice",
    "odor", "essential", "accord", "nuance"
])
final_remove_words = set()
for w in remove_words:
    final_remove_words.add(w)
    final_remove_words.add(w + "s")

num_lines = sum(1 for line in open(IMPORT_FILE, 'r'))
all_docs = []

print("Reading file")
with open(IMPORT_FILE) as f:
    for line in tqdm.tqdm(f, total=num_lines):
        if line == "\n":
            continue
        all_docs.append(line.split(",")[:-1])


def split_on_conjuction(descriptor):
    if " and " in descriptor:
        return [split_on_conjuction(sw) for sw in descriptor.split(" and ")]
    if " or " in descriptor:
        return [split_on_conjuction(sw) for sw in descriptor.split(" or ")]
    return [descriptor.strip()]


def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def remove_bad_words(descriptor):
    return " ".join([
        word for word in descriptor.split() if word not in final_remove_words
    ])


def cleanup(doc):
    doc = flatten([split_on_conjuction(desc) for desc in doc])
    doc = [remove_bad_words(desc) for desc in doc]
    return doc


print("Cleaning docs.")
for i, doc in enumerate(tqdm.tqdm(all_docs)):
    all_docs[i] = cleanup(doc)

all_words_set = set()
occurs = Counter()
for doc in tqdm.tqdm(all_docs):
    for desc in doc:
        for word in desc.split():
            all_words_set.add(word)
            occurs[word] += 1

all_words = sorted(list(all_words_set))

print("Calculating embeddings.")
all_inputs = TOKENIZER(all_words, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = MODEL(**all_inputs, output_hidden_states=True)
hidden_states = outputs["hidden_states"]
# Using last hidden layer
last_embeds = hidden_states[-1]
# Just use the [CLS] token (or <s>)
hidden_rep = last_embeds[:, 0, :]
distances = torch.cdist(hidden_rep, hidden_rep)


def calculate_rank(ratings, ascending=True):
    df = pd.Series(ratings)
    return np.array(
        df.rank(ascending=ascending, method='average').values.tolist())


def check_rankings(word, potentials, freqs, dists, fuzzes):
    fr_r = calculate_rank(freqs, ascending=False)
    di_r = calculate_rank(dists)
    fz_r = calculate_rank(fuzzes, ascending=False)
    total_r = fr_r + di_r + fz_r
    return potentials[np.argmin(total_r)]


def correct_word(w1):
    i = all_words.index(w1)
    potentials = []
    freqs = np.array([])
    dists = np.array([])
    fuzzes = np.array([])

    for j, w2 in enumerate(all_words):
        eb_d = distances[i, j]
        fz_d = fuzz.partial_ratio(w1, w2)
        ln_d = abs(len(w1) - len(w2)) / len(w1)
        if eb_d < .3 and fz_d > 90 and ln_d < .25:
            potentials.append(w2)
            freqs = np.append(freqs, occurs[w2])
            dists = np.append(dists, eb_d)
            fuzzes = np.append(fuzzes, fz_d)

    if not potentials or (len(potentials) == 1 and potentials[0] == w1):
        return w1

    return check_rankings(word, potentials, freqs, dists, fuzzes)


print("Calculating replacements")
corrections = dict()
for word in tqdm.tqdm(all_words):
    corrections[word] = correct_word(word)

corrections = {k: v for k, v in corrections.items() if k != v}

for k, v in corrections.items():
    print(f"{k} -> {v}")

print(f"Found a total of {len(corrections.items())} corrections.")


def get_correction(word):
    if word in corrections and corrections[word] != word:
        return get_correction(corrections[word])
    return word


print("Applying corrections")
all_new_docs = []
for doc in tqdm.tqdm(all_docs):
    ndoc = []
    for desc in doc:
        ndesc = []
        for word in desc.split():
            ndesc.append(get_correction(word))
        ndoc.append(" ".join(ndesc).strip())
    all_new_docs.append(ndoc)


def write_docs_to_file(all_docs, out_file):
    with open(out_file, 'wt') as outpath:
        for all_docs in tqdm.tqdm(all_docs):
            if not all_docs:
                continue

            for word in all_docs:
                outpath.write(word + WORD_SEP)

            outpath.write("\n")
            outpath.write("\n")


write_docs_to_file(all_new_docs, CORRECTED_FILE)
