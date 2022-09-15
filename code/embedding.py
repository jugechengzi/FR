import gzip

import numpy as np


def get_embeddings(embedding_path):
    with gzip.open(embedding_path, "rt") as file:
        lines = file.readlines()
    embedding_tensor = []
    word2idx = {}
    for indx, l in enumerate(lines):
        word, emb = l.split()[0], l.split()[1:]
        vector = [float(x) for x in emb]
        if indx == 0:
            embedding_tensor.append(np.zeros(len(vector)))
        embedding_tensor.append(vector)
        word2idx[word] = indx + 1
    embedding_tensor = np.array(embedding_tensor, dtype=np.float32)
    return embedding_tensor, word2idx


def get_glove_embedding(glove_embedding_path):
    with open(glove_embedding_path, 'rt',encoding='utf-8') as f:
        lines = f.readlines()
        embedding = []
        word2idx = {}
        for indx, line in enumerate(lines):
            word, emb = line.split()[0], line.split()[1:]
            vector = [float(x) for x in emb]
            if indx == 0:
                embedding.append(np.zeros(len(vector)))
            embedding.append(vector)
            word2idx[word] = indx + 1

        embedding = np.array(embedding, dtype=np.float32)

        return embedding, word2idx
