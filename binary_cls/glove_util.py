import os
import sys
import numpy as np
from tqdm import tqdm

cur_path = os.path.abspath(__file__)
cur_dir  = os.path.dirname(cur_path)
par_dir  = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(par_dir)


class Glove_Model():
    def __init__(self):
        self.word_vec = {}
        self.vocab = []

        file_path = os.path.join(par_dir, "pretrained_model/glove/glove.6B.100d.txt")
        file = open(file_path, "r")
        lines = file.readlines()
        for line in lines:
            item = line.split()
            word = item[0]
            self.vocab.append(word)
            self.word_vec[word] = np.array([float(num) for num in item[1:]])

    def word2vec(self, word):
        return self.word_vec[word]


class Word_Model():
    def __init__(self):
        unk_token = "UNK"
        pad_token = "PAD"
        self.vocab = {unk_token: 0, pad_token: 1}
        self.word_vecs = None
        self.embedding_size = 100
        self.model = Glove_Model()
        for idx, word in enumerate(self.model.vocab):
            if idx > 25000: break
            self.vocab[word] = len(self.vocab)

        self.vocab_size = len(self.vocab) + 2
        self.word_vecs = np.random.rand(self.vocab_size, self.embedding_size) * 0.2 - 0.1
        for word in tqdm(self.vocab):
            idx = self.vocab[word]
            if word in self.model.vocab:
                self.word_vecs[idx] = self.model.word_vec[word]

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return self.vocab_size

    def get_word_vecs(self):
        return self.word_vecs

    def word2index(self, word):
        if word not in self.vocab:
            return 0  # unk_token
        else:
            return self.vocab[word]
