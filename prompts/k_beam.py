# python3 -m prompts.k_beam
import util.read_dataset as dataset

dataset.read_all_words("all_lemmatized")

from odormatic import data_util
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import tqdm
import random
import numpy as np
from collections import Counter

NUM_BEAMS = 75
NUM_FINGERS = 75
LENGTH_PROMPT = 50

SWITCH_LAYER_ODDS = .15

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


def get_score(prompt, layer):
    try:
        get_embedding_fn_all_words = lambda all_descs: calculate_embedding_dictionary(
            all_descs, prompt, layer)
        s1 = data_util.get_correlation(get_embedding_fn_all_words,
                                       do_contextual=False)
        s2 = data_util.get_correlation(get_embedding_fn_all_words,
                                       do_contextual=True)
        return s1 + s2
    except IndexError:
        return 0

descs = list(dataset.occurs.keys())
probs = np.array(list(dataset.occurs.values())) / np.sum(
    list(dataset.occurs.values()))

samples = list(set(np.random.choice(descs, NUM_BEAMS * NUM_FINGERS, p=probs)))
generation = [desc + " {descriptor}" for desc in samples]
best_score = 0
best_prompt = ""
best_gen = 0
generation = [("{descriptor}", None)]
for i in range(1, LENGTH_PROMPT):
    samples = []

    # We want to generate a number of samples across the whole generation equal to
    # beams*fingers. This really only affects the first generation.
    num_fingers = int(NUM_BEAMS * NUM_FINGERS / len(generation))
    # Construct the generation
    for prompt, layer in generation:
        prefixes = list(set(np.random.choice(descs, num_fingers, p=probs)))
        for prefix in prefixes:
            if not layer:
                layer = random.randrange(0, 24)

            if random.random() < SWITCH_LAYER_ODDS:
                layer += 1

            if random.random() < SWITCH_LAYER_ODDS:
                layer -= 1
            # Either attach as a prefix or a suffix
            if random.random() < .5:
                samples.append((prefix + " " + prompt, layer))
            else:
                samples.append((prompt + " " + prefix, layer))

    # Evaluate all samples
    scores = Counter()
    for prompt, layer in tqdm.tqdm(samples):
        score = get_score(prompt, layer)
        scores[(prompt, layer)] = score
        if score > best_score:
            best_score = score
            best_prompt = (prompt, layer)
            best_gen = i

    # Save to file
    with open(f"prompts/generated/k_beam{i}.csv", 'w') as f:
        for k,v in scores.most_common():
            f.write( "{}\t\t\t{}\n".format(k,v) )

    # Pick the top fingers as next generation
    print("TOP TEN")
    print(scores.most_common(10))
    generation = [prompt for prompt, score in scores.most_common(NUM_BEAMS)]
    print(
        f"Finished generation {i}.\nBest prompt was {best_prompt} from generation {best_gen} with score {best_score}."
    )

