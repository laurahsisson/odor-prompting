import torch
import random
from collections import Counter
import string
import tqdm

import time
import json

import numpy as np

WORD_SEP = ", "

all_docs = []
all_words_set = set()
all_descs = []
cooccur = Counter()
occurs = Counter()


def with_shape(t, shape):
    assert t.shape == shape
    return t


def get_word_pair(w1, w2):
    if w1 < w2:
        return w1 + WORD_SEP + w2
    else:
        return w2 + WORD_SEP + w1


def get_cooccurence_matrix(word_batch):
    bsz = len(word_batch)
    cooccur_matrix = []
    for i, word1 in enumerate(word_batch):
        cooccur_matrix.append([])

        for word2 in word_batch:
            pair = get_word_pair(word1, word2)
            # Calculate pointwise mutual information, with small epsilon to avoid divide by zero
            cc = cooccur[pair] if cooccur[pair] != 0 else 1e-8
            pmi = np.log(cc / (occurs[word1] * occurs[word2]))
            # Convert to distance
            cooccur_matrix[i].append(-1 * pmi)

    return torch.FloatTensor(cooccur_matrix)


def get_weight_matrix(word_batch):
    return torch.FloatTensor(
        [[occurs[w1] * occurs[w2] if w1 != w2 else 0 for w2 in word_batch]
         for w1 in word_batch])


def get_word_weight(word_batch):
    return torch.FloatTensor([occurs[w] for w in word_batch])


def read_all_words(dataset):
    # Unclean solution, but fine for now.
    global all_docs, all_words_set, all_descs, cooccur, occurs

    with open(f"data/{dataset}.txt") as f:
        current_doc = []
        for line in f:
            if line == "\n":
                all_docs.append(current_doc)
                current_doc = []
            else:
                for w_raw in line.split(WORD_SEP):
                    word = w_raw.strip()
                    if word and word not in current_doc:
                        current_doc.append(word)

    for doc in all_docs:
        for i, word1 in enumerate(doc):
            for word2 in doc[i:]:
                pair = get_word_pair(word1, word2)
                cooccur[pair] += 1

            occurs[word1] += 1
            if word1 not in all_words_set:
                all_descs.append(word1)
                all_words_set.add(word1)