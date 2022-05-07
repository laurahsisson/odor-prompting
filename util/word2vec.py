import tqdm
import numpy as np
from sklearn.cluster import KMeans

FULLWORD_FILE = "data/wiki-news-300d-1M.vec"
SUBWORD_FILE = "wiki-news-300d-1M-subword.vec"
NUM_WORDS = 0
EMBED_SIZE = 0

N_CLUSTERS = 25


def parse_header(line):
    num_words, embed_size = line.split()
    return int(num_words), int(embed_size)

# Reads the specified words from the given w2v file. If no words are specified, then reads all words.
def read_w2v_file(w2v_file, all_words=set()):
    global NUM_WORDS
    global EMBED_SIZE

    all_embeds = dict()

    print(f"Reading from {w2v_file}")
    num_lines = sum(1 for line in open(w2v_file, 'r'))
    with open(w2v_file) as f:
        all_lines = f.readlines()
        NUM_WORDS, EMBED_SIZE = parse_header(all_lines[0])
        for line in tqdm.tqdm(all_lines[1:]):
            splits = line.split()
            word = splits[0]

            if len(all_words) > 0 and not word in all_words:
                continue

            all_embeds[word] = np.array(splits[1:]).astype(float)

    return all_embeds


def get_fullword_embeddings(all_words):
    return read_w2v_file(FULLWORD_FILE, all_words)