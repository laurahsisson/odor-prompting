import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from fuzzywuzzy import fuzz, process
import tqdm
from collections import Counter
import numpy as np
import pandas as pd

MODEL_NAME = "distilroberta-base"
TEST_FILE = "test_word_list.txt"

MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

test_pairs = []
good_words = set()
all_words_set = set()

remove_words = set([
    "a", "an", "the", "essence", "oil", "like", "undertone", "note", "juice",
    "odor", "essential"
])
final_remove_words = set()
for w in remove_words:
    final_remove_words.add(w)
    final_remove_words.add(w + "s")

with open(TEST_FILE) as f:
    for line in f:
        inp, cor = line.strip().split(" -> ")
        if "?" in cor or "and" in cor or "or" in cor:
            print(f"Skipping {inp} -> {cor}")
            continue

        test_pairs.append((inp, cor))
        for w in inp.split():
            all_words_set.add(w)

        for w in cor.split():
            all_words_set.add(w)
            good_words.add(w)

IMPORT_FILE = "../data/all_lemmatized.txt"

num_lines = sum(1 for line in open(IMPORT_FILE, 'r'))
all_docs = []

with open(IMPORT_FILE) as f:
    for line in tqdm.tqdm(f, total=num_lines):
        if line == "\n":
            continue
        to_add = []
        doc_words = line.split(",")[:-1]
        for w in doc_words:
            word = w.strip()
            if word in to_add:
                continue
            to_add.append(word)
        all_docs.append(to_add)

occurs = Counter()
for d in all_docs:
    for c in d:
        occurs[c] += 1

all_words = list(all_words_set)
all_inputs = TOKENIZER(all_words, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = MODEL(**all_inputs, output_hidden_states=True)
hidden_states = outputs["hidden_states"]
# Using last hidden layer
last_embeds = hidden_states[-1]
# Just use the [CLS] token (or <s>)
hidden_rep = last_embeds[:, 0, :]
# It is actually very expensive to calculate cdist across the whole thing, so we might just use fuzzy matcher first and then go by there.
# Regardless, we can just use cdist here lol
distances = torch.cdist(hidden_rep, hidden_rep)

print("Calculating replacements")

# for i, w1 in enumerate(all_words):
# 	for j, w2 in enumerate(all_words):
# 		if j <= i:
# 			continue
# 		if distances[i,j] < .3:
# 			print(w1,w2)


def calculate_rank(ratings, ascending=True):
    df = pd.Series(ratings)
    return np.array(
        df.rank(ascending=ascending, method='average').values.tolist())


def check_rankings(potentials, freqs, dists, fuzzes):
    fr_r = calculate_rank(freqs, ascending=False)
    di_r = calculate_rank(dists)
    fz_r = calculate_rank(fuzzes)
    total_r = fr_r + di_r + fz_r
    return potentials[np.argmin(total_r)]


def correct_word(w1):
    if w1 in remove_words:
        return ""
    # Probably have to sort this in production
    i = all_words.index(w1)
    potentials = []
    freqs = np.array([])
    dists = np.array([])
    fuzzes = np.array([])

    for j, w2 in enumerate(all_words):
        eb_d = distances[i, j]
        fz_d = fuzz.partial_ratio(w1, w2)
        if eb_d < .3 and fz_d > 90:
            potentials.append(w2)
            freqs = np.append(freqs, occurs[w2])
            dists = np.append(dists, eb_d)
            fuzzes = np.append(fuzzes, fz_d)

    if not potentials:
        return w1

    return check_rankings(potentials, freqs, dists, fuzzes)

    # ?    return process.extractOne(w1, potentials)[0]
    # No corrections were found

    # # Find the closest one
    # print(f"{w1} ... {potentials}")
    # best = min(potentials,key=lambda p: p[1])
    return max(potentials, key=lambda w: occurs[w])


for inp, cor in test_pairs:
    res = [correct_word(w) for w in inp.split()]
    res = [w for w in res if w != ""]
    res = " ".join(res)

    if res == cor:
        continue

    print(inp, "_____", res.encode(), "_____", cor.encode())

# Basically replace with this. Also check edit distance. Replace with more freq.

# for inp, cor in test_pairs:
#     for w1 in inp.split():
#         for w2 in cor.split():
#             print(w1,w2,fuzz.ratio(w1,w2))
