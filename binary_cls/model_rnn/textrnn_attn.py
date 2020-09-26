import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class RNN_Attn_CLS(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_labels=2):
        super(RNN_Attn_CLS, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)

        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=False,
                           dropout=0.1,
                           bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, num_labels)

    def attention_net(self, rnn_output, final_state):
        # hidden:     [batch_size, 2 * hidden_size, 1]
        # rnn_output: [batch_size, seq_len, 2 * hidden_size]
        hidden = final_state.view(-1, 2 * self.hidden_size, 1)
        # attn_weights: [batch_size, seq_len, 1]
        attn_weights = torch.bmm(rnn_output, hidden)
        # attn_weights: [batch_size, seq_len]
        attn_weights = attn_weights.squeeze(2)

        # soft_attn_weights: [batch_size, seq_len]
        soft_attn_weights = F.softmax(attn_weights, dim=1)

        # rnn_output: [batch_size, 2 * hidden_size, seq_len]
        # soft_attn_weights: [batch_size, seq_len, 1]
        # context:    [batch_size, 2 * hidden_size, 1]
        # context:    [batch_size, 2 * hidden_size]
        rnn_output = rnn_output.transpose(1, 2)
        soft_attn_weights = soft_attn_weights.unsqueeze(2)
        context = torch.bmm(rnn_output, soft_attn_weights)
        context = context.squeeze(2)

        return context, soft_attn_weights

    # eval_accu: 0.8279816513761468
    def forward(self, input_ids, input_mask, labels):
        batch_size = input_ids.size(0)
        input = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        input = input.permute(1, 0, 2)     # [seq_len, batch_size, hidden_size]

        # hidden_state: [2, batch_size, hidden_size]
        # cell_state:   [2, batch_size, hidden_size]
        hidden_state = torch.zeros(1*2, batch_size, self.hidden_size).cuda()
        cell_state   = torch.zeros(1*2, batch_size, self.hidden_size).cuda()

        # rnn_output: [seq_len, batch_size, hidden_size]
        # rnn_output: [batch_size, seq_len, hidden_size]
        rnn_output, (final_hidden_state, final_cell_state) = self.rnn(input, (hidden_state, cell_state))
        rnn_output = rnn_output.permute(1, 0, 2)

        attn_output, attention = self.attention_net(rnn_output, final_hidden_state)

        logits = self.linear(attn_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits