import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class RCNN_CLS(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_labels=2):
        super(RCNN_CLS, self).__init__()
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.1)
        self.maxpool = nn.MaxPool1d(32)
        self.linear  = nn.Linear(2 * hidden_size + embedding_size, num_labels)

    # eval_accu: 0.8348623853211009
    def forward(self, input_ids, input_mask, labels):
        rnn_input = self.embedding(input_ids)
        rnn_output, _ = self.lstm(rnn_input)
        rnn_output = torch.cat((rnn_input, rnn_output), dim=2)
        rnn_output = F.relu(rnn_output)
        rnn_output = rnn_output.permute(0, 2, 1)
        cnn_output = self.maxpool(rnn_output).squeeze()
        logits = self.linear(cnn_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


