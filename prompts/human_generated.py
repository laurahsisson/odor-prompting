# python3 -m prompts.test_human_generated
from odormatic import data_util
import tqdm
import random

from util import embedding


prompt = "essense {descriptor} flavored"

with open("prompts/generated/human_generated.csv", 'w') as f:
    f.write("Prompt,Layer,Score\n")

    for layer in tqdm.tqdm(range(23)):
        get_embedding_fn_all_words = lambda all_descs: embedding.calculate_embedding_dictionary(
            all_descs, prompt, layer)
        score1 = data_util.get_correlation(get_embedding_fn_all_words,
                                          do_contextual=False)
        score2 = data_util.get_correlation(get_embedding_fn_all_words,
                                  do_contextual=True)
        score = (score1+score2)
        f.write(f"{prompt},{layer},{(score1+score2)/2},{score1},{score2}\n")

        if score > best_score:
            best_score = score
            best_data = (prompt,layer,score1,score2)
            print(best_score,best_data)

