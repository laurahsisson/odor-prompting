# python3 -m charts.layer_performance
from odormatic import data_util
from embedding import util

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import tqdm
import random
import numpy as np
import pandas as pd
from collections import Counter
from util import word2vec as w2v
import matplotlib.pyplot as plt

MODEL_NAME = "bert-large-uncased"
MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

NUM_LAYERS = 25


def calculate_embedding(all_descs, prompt, layer):
    all_prompts = [prompt.format(descriptor=desc) for desc in all_descs]
    all_inputs = TOKENIZER(all_prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = MODEL(**all_inputs, output_hidden_states=True)
    hidden_states = outputs["hidden_states"]
    # Using layer -11 (from previous experiments)
    embeds = hidden_states[layer]
    # Ignore [CLS] and [SEP] tokens
    hidden_rep = embeds[:, :, :]
    # Calculate mean across the remaining tokens
    return hidden_rep.mean(dim=1)


def calculate_embedding_dictionary(all_descs, prompt, layer):
    embed_tensor = calculate_embedding(all_descs, prompt, layer).cpu().numpy()
    embeddings = dict()
    for i, tensor in enumerate(embed_tensor):
        embeddings[all_descs[i]] = tensor
    return embeddings


singlewords = []
fulldescriptors = []

for i in tqdm.tqdm(range(NUM_LAYERS)):
	get_embedding_fn_all_words = lambda all_descs: calculate_embedding_dictionary(
	    all_descs, "{descriptor}", i)
	singlewords.append(data_util.get_correlation(get_embedding_fn_all_words,
                                        do_contextual=False))
	fulldescriptors.append(data_util.get_correlation(get_embedding_fn_all_words,
                                        do_contextual=True))

fig, ax = plt.subplots()
ax.set_xlabel('hidden layer')
ax.set_ylabel('Pearson correlation coefficient')

#define aesthetics for plot
color1 = 'steelblue'
color2 = 'red'
line_size = 1
marker_size = 5

ax.plot(list(range(NUM_LAYERS)),
        singlewords,
        color=color1,
        marker="x",
        ms=marker_size,
        lw=line_size,
        label="single-word")
ax.plot(list(range(NUM_LAYERS)),
        fulldescriptors,
        color=color2,
        marker="x",
        ms=marker_size,
        lw=line_size,
        label="full-descriptor")
ax.legend()


plt.savefig("figures/layer_performance.png")
plt.savefig("figures/layer_performance.svg")