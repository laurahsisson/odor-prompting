# python3 -m prompts.random_score
import numpy as np
from odormatic import data_util
import tqdm

scores = []
for i in tqdm.tqdm(range(100)):
    get_embedding_fn_all_words = lambda all_descs: {
        desc: np.random.rand(1024)
        for desc in all_descs
    }
    final_acc = data_util.get_correlation(get_embedding_fn_all_words,
                                          do_contextual=True)
    scores.append(final_acc)

print(np.mean(scores))
