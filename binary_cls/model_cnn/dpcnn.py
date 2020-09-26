# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class Model(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, num_labels=2):
        super(Model, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)

        self.num_filters = 80
        self.conv_region = nn.Conv2d(1, self.num_filters, (3, embedding_size), stride=1)
        self.conv = nn.Conv2d(self.num_filters, self.num_filters, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom

        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.num_filters, num_labels)

    def forward(self, input_ids, input_mask, labels):
        input = self.embedding(input_ids)
        input = input.unsqueeze(1)

        input = self.conv_region(input)

        input = self.padding1(input)
        input = self.relu(input)
        input = self.conv(input)

        input = self.padding2(input)
        input = self.relu(input)
        input = self.conv(input)

        while input.size()[2] > 2:
            input = self._block(input)
        input = input.squeeze()
        logits = self.linear(input)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        x = x + px
        return x
