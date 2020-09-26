import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class Fasttext(nn.Module):
    def __init__(self, embeddings, embedding_size, num_labels=2):
        super(Fasttext, self).__init__()
        self.n_gram_vocab = 250499
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)
        self.embedding_ngram2 = nn.Embedding(self.n_gram_vocab, embedding_size)
        self.embedding_ngram3 = nn.Embedding(self.n_gram_vocab, embedding_size)
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(3*embedding_size, embedding_size)
        self.linear2 = nn.Linear(embedding_size, num_labels)

    def forward(self, input_ids, input_ids_gram2, input_ids_gram3, input_mask, labels):
        out_word    = self.embedding(input_ids)
        out_bigram  = self.embedding_ngram2(input_ids_gram2)
        out_trigram = self.embedding_ngram3(input_ids_gram3)

        out = torch.cat((out_word, out_bigram, out_trigram), dim=-1)

        out = out.mean(dim=-1)
        out = self.dropout(out)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return out
