import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class RNN_CLS(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_labels=2):
        super(RNN_CLS, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)

        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True,
                          dropout=0.1,
                          bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, num_labels)

    # # 定长RNN
    # def forward(self, input_ids, input_mask, labels):
    #     rnn_input = self.embedding(input_ids)
    #     rnn_output, hidden_output = self.rnn(rnn_input)
    #     rnn_output = rnn_output[:, -1, :]
    #     logits = self.linear(rnn_output)
    #
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits, labels)
    #         return loss, logits
    #     else:
    #         return logits

    # # 变长RNN
    # def forward(self, input_ids, input_mask, labels):
    #     batch_size = input_ids.size(0)
    #     rnn_input = self.embedding(input_ids)
    #     lengths = input_mask.eq(1).long().sum(1).squeeze()
    #
    #     rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True, enforce_sorted=False)
    #     rnn_output, hidden_output = self.rnn(rnn_input)
    #     rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    #
    #     # hidden_output[-2, :, : ] is the last of the forwards RNN
    #     # hidden_output[-1, :, : ] is the last of the backwards RNN
    #     hidden_output = torch.cat((hidden_output[-2, :, :], hidden_output[-1, :, :]), dim=1)
    #     logits = self.linear(hidden_output)
    #
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits, labels)
    #         return loss, logits
    #     else:
    #         return logits

    # 变长RNN (对输入按长度排序)
    def forward(self, input_ids, input_mask, labels):
        batch_size = input_ids.size(0)
        rnn_input = self.embedding(input_ids)
        lengths = input_mask.eq(1).long().sum(1).squeeze()

        # idx_sort: 长度由大到小排序后的索引
        # idx_unsort: 排好序的长度对应回原长度的索引
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)  # 排序后，原序列的index

        lengths = list(lengths[idx_sort])
        rnn_input = torch.index_select(rnn_input, 0, idx_sort)  # 0表示按行索引

        # hidden_output[-2, :, : ] is the last of the forwards RNN
        # hidden_output[-1, :, : ] is the last of the backwards RNN
        rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True)
        rnn_output, hidden_output = self.rnn(rnn_input)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        hidden_output = torch.cat((hidden_output[-2, :, :], hidden_output[-1, :, :]), dim=1)

        rnn_output = torch.index_select(rnn_output, 0, idx_unsort)  # 0表示按行索引
        hidden_output = torch.index_select(hidden_output, 0, idx_unsort)  # 0表示按行索引
        logits = self.linear(hidden_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class LSTM_CLS(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_labels=2):
        super(LSTM_CLS, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)

        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=hidden_size,
                           num_layers=1,
                           batch_first=True,
                           dropout=0.1,
                           bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, num_labels)

    # # 定长LSTM
    # def forward(self, input_ids, input_mask, labels):
    #     rnn_input = self.embedding(input_ids)
    #     rnn_output, _ = self.rnn(rnn_input)
    #     rnn_output = rnn_output[:, -1, :]  # 句子最后时刻的 hidden state
    #     logits = self.linear(rnn_output)
    #
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits, labels)
    #         return loss, logits
    #     else:
    #         return logits

    # # 变长LSTM
    # eval_accu: 0.8314220183486238
    # def forward(self, input_ids, input_mask, labels):
    #     batch_size = input_ids.size(0)
    #     rnn_input = self.embedding(input_ids)
    #     lengths = input_mask.eq(1).long().sum(1).squeeze()
    #
    #     rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True, enforce_sorted=False)
    #     rnn_output, (hidden_output, cell_output) = self.rnn(rnn_input)
    #     rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    #
    #     # hidden_output[-2, :, : ] is the last of the forwards RNN
    #     # hidden_output[-1, :, : ] is the last of the backwards RNN
    #     hidden_output = torch.cat((hidden_output[-2, :, :], hidden_output[-1, :, :]), dim=1)
    #     logits = self.linear(hidden_output)
    #
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits, labels)
    #         return loss, logits
    #     else:
    #         return logits

    # 变长LSTM (对输入按长度排序)
    # eval_accu: 0.83371559633027530
    def forward(self, input_ids, input_mask, labels):
        batch_size = input_ids.size(0)
        rnn_input = self.embedding(input_ids)
        lengths = input_mask.eq(1).long().sum(1).squeeze()

        # idx_sort: 长度由大到小排序后的索引
        # idx_unsort: 排好序的长度对应回原长度的索引
        _, idx_sort   = torch.sort(lengths,  dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)  # 排序后，原序列的index

        lengths = list(lengths[idx_sort])
        rnn_input = torch.index_select(rnn_input, 0, idx_sort)  # 0表示按行索引

        # hidden_output[-2, :, : ] is the last of the forwards RNN
        # hidden_output[-1, :, : ] is the last of the backwards RNN
        rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True)
        rnn_output, (hidden_output, cell_output) = self.rnn(rnn_input)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
        hidden_output = torch.cat((hidden_output[-2, :, :], hidden_output[-1, :, :]), dim=1)

        rnn_output    = torch.index_select(rnn_output,    0, idx_unsort)  # 0表示按行索引
        hidden_output = torch.index_select(hidden_output, 0, idx_unsort)  # 0表示按行索引
        logits = self.linear(hidden_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class GRU_CLS(nn.Module):
    def __init__(self, embeddings, vocab_size, embedding_size, hidden_size, num_labels=2):
        super(GRU_CLS, self).__init__()
        self.hidden_size = hidden_size
        # self.embedding = nn.Embedding(vocab_size, embedding_size)
        # self.embedding = nn.Embedding(vocab_size, embedding_size, _weight=torch.from_numpy(embeddings).float())
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float(), freeze=False)

        self.rnn = nn.GRU(input_size=embedding_size,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True,
                          dropout=0.1,
                          bidirectional=True)
        self.linear = nn.Linear(hidden_size, num_labels)

    # # 定长GRU
    # # eval_accu: 0.841743119266055
    # def forward(self, input_ids, input_mask, labels):
    #     rnn_input = self.embedding(input_ids)
    #     rnn_output, _ = self.rnn(rnn_input)
    #     rnn_output = rnn_output[:, -1, :]
    #     logits = self.linear(rnn_output)
    #
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits, labels)
    #         return loss, logits
    #     else:
    #         return logits

    # # 变长GRU
    # # eval_accu: 0.8463302752293578
    # def forward(self, input_ids, input_mask, labels):
    #     batch_size = input_ids.size(0)
    #     rnn_input = self.embedding(input_ids)
    #     lengths = input_mask.eq(1).long().sum(1).squeeze()
    #
    #     rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True, enforce_sorted=False)
    #     rnn_output, (hidden_output, cell_output) = self.rnn(rnn_input)
    #     rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)
    #
    #     logits = self.linear(hidden_output)
    #
    #     if labels is not None:
    #         loss_fct = CrossEntropyLoss()
    #         loss = loss_fct(logits, labels)
    #         return loss, logits
    #     else:
    #         return logits

    # 变长GRU (对输入按长度排序)
    # eval_accu: 0.8463302752293578
    def forward(self, input_ids, input_mask, labels):
        batch_size = input_ids.size(0)
        rnn_input = self.embedding(input_ids)
        lengths = input_mask.eq(1).long().sum(1).squeeze()

        # idx_sort: 长度由大到小排序后的索引
        # idx_unsort: 排好序的长度对应回原长度的索引
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)  # 排序后，原序列的index

        lengths = list(lengths[idx_sort])
        rnn_input = torch.index_select(rnn_input, 0, idx_sort)  # 0表示按行索引

        # hidden_output[-2, :, : ] is the last of the forwards RNN
        # hidden_output[-1, :, : ] is the last of the backwards RNN
        rnn_input = nn.utils.rnn.pack_padded_sequence(rnn_input, lengths, batch_first=True)
        rnn_output, (hidden_output, cell_output) = self.rnn(rnn_input)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        rnn_output    = torch.index_select(rnn_output,    0, idx_unsort)  # 0表示按行索引
        hidden_output = torch.index_select(hidden_output, 0, idx_unsort)  # 0表示按行索引

        logits = self.linear(hidden_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
