# python3 -m charts.score_by_model
from odormatic import data_util
import random
import numpy as np
import pandas as pd

from util import word2vec
from util import embedding


def get_score(task, model, return_score_by_descriptor):
    if task == "contextual" and model == "DirSem":
        return 0

    get_embedding_fn_all_words = lambda all_descs: {
        desc: [0]
        for desc in all_descs
    }
    if model == "Random":
        get_embedding_fn_all_words = lambda all_descs: {
            desc: np.random.rand(1024)
            for desc in all_descs
        }
    if model == "BERT":
        get_embedding_fn_all_words = lambda all_descs: embedding.calculate_embedding_dictionary(
            all_descs, "{descriptor}", 10)
    if model == "HumanPrompt":
        get_embedding_fn_all_words = lambda all_descs: embedding.calculate_embedding_dictionary(
            all_descs, "essence {descriptor} flavored", 12)
    if model == "MinedPrompt":
        get_embedding_fn_all_words = lambda all_descs: embedding.calculate_embedding_dictionary(
            all_descs,
            "many woody fresh butter popcorn woody orange biscuit rubbery {descriptor} intense garlic woody comments warm evaluation comments umami grape rain one",
            11)
    if model == "DirSem":
        get_embedding_fn_all_words = lambda all_descs: word2vec.get_fullword_embeddings(
            all_descs)
    if model == "WordPiece":
        get_embedding_fn_all_words = lambda all_descs: embedding.calculate_wordpiece_embedding(
            all_descs)

    print(task, model)
    if task == "single-word":
        return data_util.get_correlation(
            get_embedding_fn_all_words,
            do_contextual=False,
            return_score_by_descriptor=return_score_by_descriptor)
    if task == "full-descriptor":
        return data_util.get_correlation(
            get_embedding_fn_all_words,
            do_contextual=True,
            return_score_by_descriptor=return_score_by_descriptor)


print("Correlation coefficient by model")
tasks = ["single-word", "full-descriptor"]
models = [
    "WordPiece", "DirSem", "BERT", "MinedPrompt", "HumanPrompt", "Random"
]
all_data = [[get_score(task, model, False) for task in tasks]
            for model in models]
df = pd.DataFrame(all_data, columns=tasks, index=models)
df = df.T
df.to_csv("figures/score_by_model.csv")
print(df)

print()
print("Correlation coefficient in best models by descriptor")
previous, descriptors = get_score("single-word", "DirSem", True)
best, _ = get_score("single-word", "MinedPrompt", True)
all_data = [previous, best, np.subtract(best, previous)]
df = pd.DataFrame(all_data,
                  columns=descriptors,
                  index=["DirSem", "MinedPrompt", "Improvement"])
df = df.T
df.to_csv("figures/improvement_by_desc.csv")
print(df)
