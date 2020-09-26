# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class TextCNN(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, max_seq_len=32, label_num=2):
        super(TextCNN, self).__init__()
        self.label_num = label_num
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)

        self.conv3 = nn.Conv2d(1, 1, (3, embedding_size))
        self.conv4 = nn.Conv2d(1, 1, (4, embedding_size))
        self.conv5 = nn.Conv2d(1, 1, (5, embedding_size))
        self.Max3_pool = nn.MaxPool2d((max_seq_len-3+1, 1))
        self.Max4_pool = nn.MaxPool2d((max_seq_len-4+1, 1))
        self.Max5_pool = nn.MaxPool2d((max_seq_len-5+1, 1))
        self.dropout = nn.Dropout(0.1)
        self.linear  = nn.Linear(3, label_num)

    def forward(self, input_ids, input_mask, labels):
        x = self.embedding(input_ids)
        batch = x.shape[0]

        x = x.unsqueeze(1)
        # Convolution
        x1 = F.relu(self.conv3(x))
        x2 = F.relu(self.conv4(x))
        x3 = F.relu(self.conv5(x))

        # Pooling
        x1 = self.Max3_pool(x1)
        x2 = self.Max4_pool(x2)
        x3 = self.Max5_pool(x3)

        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1)
        x = x.view(batch, 1, -1)

        # project the features to the labels
        x = self.dropout(x)
        x = self.linear(x)
        x = x.view(-1, self.label_num)

        logits = x
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
