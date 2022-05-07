# python3 -m charts.pca_embedding
from odormatic import data_util
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import tqdm
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from util import word2vec as w2v

import numpy as np

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt

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
    return hidden_rep.mean(dim=1).numpy()


def calculate_embedding_w2v(all_descs):
    embedding_dictionary = w2v.get_fullword_embeddings(all_descs)
    all_embeddings = []
    for desc in all_descs:
        all_embeddings.append(embedding_dictionary[desc])
    return np.stack(all_embeddings)


drv_words = [
    'citrus', 'lemon', 'grapefruit', 'orange', 'fruity', 'pineapple', 'grape',
    'strawberry', 'apple', 'pear', 'cantaloupe', 'peach', 'banana', 'floral',
    'rose', 'violets', 'lavender', 'cologne', 'musk', 'perfumery', 'fragrant',
    'aromatic', 'honey', 'cherry', 'almond', 'acetone', 'nutty', 'spicy',
    'clove', 'cinnamon', 'laurel', 'tea', 'seasoning', 'pepper', 'peppers',
    'dill', 'caraway', 'cognac', 'resinous', 'cedarwood', 'mothballs',
    'peppermint', 'camphor', 'eucalyptus', 'chocolate', 'vanilla', 'sweet',
    'maple', 'caramel', 'malty', 'raisins', 'molasses', 'coconut', 'anise',
    'alcoholic', 'gasoline', 'turpentine', 'geranium', 'celery', 'weeds',
    'grass', 'herbal', 'cucumber', 'hay', 'grainy', 'yeasty', 'bakery',
    'fermented', 'beery', 'soapy', 'leather', 'cardboard', 'rope', 'stale',
    'musty', 'potatoes', 'mouse', 'mushroom', 'peanuts', 'beany', 'eggy',
    'bark', 'cork', 'smoky', 'incense', 'coffee', 'tar', 'creosote',
    'carbolic', 'medicinal', 'chemical', 'bitter', 'pungent', 'vinegar',
    'sauerkraut', 'ammonia', 'urine', 'urine', 'fishy', 'kipper', 'semen',
    'rubber', 'sooty', 'kerosene', 'oily', 'buttery', 'paint', 'varnish',
    'popcorn', 'meaty', 'soupy', 'rancid', 'sweaty', 'cheesy', 'methane',
    'sulfurous', 'garlic', 'metallic', 'blood', 'animal', 'sewer', 'putrid',
    'fecal', 'cadaverous', 'sickening', 'powdery', 'chalky', 'light', 'heavy',
    'cooling', 'warm'
]
drm_words = [
    'bakery', 'sweet', 'fruit', 'fish', 'garlic', 'spices', 'cold', 'sour',
    'burnt', 'acid', 'warm', 'musky', 'sweaty', 'ammonia', 'decayed', 'wood',
    'grass', 'flower', 'chemical'
]
negatives = ["jacket", "rugged", "hide", "material", "tanning"]

combined_words = drv_words + drm_words + negatives + ["amber"]

positives = {"musky", 'gasoline', "smoky", "amber", "musk"}


def get_alpha(word):
    if word == "leather":
        return 1
    if word in negatives:
        return 1
    if word in positives:
        return 1
    return .5


def get_color(word):
    if word == "leather":
        return "green"
    if word in negatives:
        return "orange"
    if word in positives:
        return "blue"
    return "grey"


def get_label(word):
    if word == "leather":
        return word
    if word in negatives:
        return word
    if word in positives:
        return word
    return ""


def calculate_distance_from_embeddings_to_centroid(embeddings, centroid):
    return np.mean(np.linalg.norm((embeddings - centroid), axis=1))


def print_triplet_statistics(all_embeddings, filename):
    negative_embeddings = []
    positive_embeddings = []
    leather_embedding = None

    for i, word in enumerate(combined_words):
        if word in negatives:
            negative_embeddings.append(all_embeddings[i])
        if word in positives:
            positive_embeddings.append(all_embeddings[i])
        if word == "leather":
            leather_embedding = all_embeddings[i]

    distance_to_centroid = calculate_distance_from_embeddings_to_centroid(
        all_embeddings, all_embeddings.mean(axis=0))

    assert leather_embedding.any()
    leather_embedding = leather_embedding

    assert len(negative_embeddings) == len(negatives)
    negative_embeddings = np.stack(negative_embeddings)
    negative_distance = calculate_distance_from_embeddings_to_centroid(
        negative_embeddings, leather_embedding)

    assert len(positive_embeddings) == len(positives)
    positive_embeddings = np.stack(positive_embeddings)
    positive_distance = calculate_distance_from_embeddings_to_centroid(
        positive_embeddings, leather_embedding)

    print(filename)
    print("Distance to centroid", distance_to_centroid)
    print("Negative distances", negative_distance)
    print("Positive distances", positive_distance)
    print("Delta",
          (negative_distance - positive_distance))
    print()


def generate_embedding_diagram(get_embedding_fn_all_words, filename):
    all_embeddings = get_embedding_fn_all_words(combined_words)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    data = {
        "Name": combined_words,
        "PC1": [x for x, y in reduced_embeddings],
        "PC2": [y for x, y in reduced_embeddings],
        "Label": [get_label(word) for word in combined_words],
        "Color": [get_color(word) for word in combined_words],
        "Opacity": [get_alpha(word) for word in combined_words]
    }

    plt.scatter(data["PC1"],
                data["PC2"],
                c=data["Color"],
                alpha=data["Opacity"])

    # for i in range(len(combined_words)):
    #     plt.annotate(data["Label"][i],xy=reduced_embeddings[i])

    plt.title(f"Embedding space for {filename.replace('_',' ')}")
    plt.xlabel("PCA_1")
    plt.ylabel("PCA_2")

    print_triplet_statistics(all_embeddings, filename)
    plt.savefig(f"figures/{filename}.svg")
    plt.savefig(f"figures/{filename}.png")


get_embedding_fn_all_words = lambda all_descs: calculate_embedding_w2v(
    all_descs)

generate_embedding_diagram(get_embedding_fn_all_words, "fasttext")

get_embedding_fn_all_words = lambda all_descs: calculate_embedding(
    combined_words, "{descriptor}", 6)

generate_embedding_diagram(get_embedding_fn_all_words, "BERT_unprompted")

get_embedding_fn_all_words = lambda all_descs: calculate_embedding(
    combined_words,
    "many woody fresh butter popcorn woody orange biscuit rubbery {descriptor} intense garlic woody comments warm evaluation comments umami grape rain one",
    11)

generate_embedding_diagram(get_embedding_fn_all_words, "BERT_mined_prompt")