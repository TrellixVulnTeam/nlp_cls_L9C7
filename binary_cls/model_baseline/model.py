import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class NGram_CLS(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, num_labels=2):
        super(NGram_CLS, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)
        self.linear = nn.Linear(embedding_size, num_labels)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, input_ids, labels):
        output = self.embedding(input_ids)
        output = output[:, 0, :]
        logits = self.linear(output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
