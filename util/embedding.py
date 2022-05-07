from odormatic import data_util
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import tqdm
import random
import numpy as np
import pandas as pd
from collections import Counter
from util import word2vec as w2v

MODEL_NAME = "bert-large-uncased"
MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)


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


def calculate_wordpiece_embedding(all_descs):
    all_inputs = TOKENIZER(all_descs, return_tensors="pt", padding=True)
    with torch.no_grad():
        embed_tensor = MODEL.base_model.embeddings.word_embeddings(
            all_inputs["input_ids"])
        embed_tensor = embed_tensor.mean(dim=1).numpy()
    embeddings = dict()
    for i, tensor in enumerate(embed_tensor):
        embeddings[all_descs[i]] = tensor
    return embeddings
