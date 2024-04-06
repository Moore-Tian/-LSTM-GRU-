import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_LM(nn.Module):
    def __init__(self, vocab_size, dim_emb, hidden_size, num_layers, limit, device=None):
        super(LSTM_LM, self).__init__()
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.limit = limit
        self.device = device
        if self.device is not None:
            self.embedding = nn.Embedding(vocab_size, dim_emb).to(self.device)
            self.lstm = nn.LSTM(dim_emb, hidden_size, num_layers, batch_first=True).to(self.device)
            self.fc = nn.Linear(hidden_size, vocab_size).to(self.device)
        else:
            self.embedding = nn.Embedding(vocab_size, dim_emb)
            self.lstm = nn.LSTM(dim_emb, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # 记一个batch内的最长序列长度为 L
        #   batch_size 为 B
        #   词汇表大小为 V
        #   隐藏层维数为 H
        #   词嵌入维度为 E
        #   x = [B, L]

        # limit = B
        limit = x.size(1)

        if self.device is not None:
            out = torch.ones(x.size(0), limit).to(self.device)
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        else:
            out = torch.ones(x.size(0), limit)
            h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # x = [B, L, E]
        emb_x = self.embedding(x)
        # output = [B, L, H]
        output, _ = self.lstm(emb_x, (h, c))
        # output = [B, L, V]
        output = self.fc(output)
        # prob = [B, L, V] （但是是按行的概率分布）
        prob = F.softmax(output, dim=2)
        # out = [B, L]，其中 index = [B, L, 1]
        out = torch.gather(prob, dim=2, index=x.unsqueeze(2)).squeeze(2)
        return out

    def loss(self, out):
        return - torch.sum(torch.log(out)) / out.size(0)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    # 这里只实现无batch的生成函数，带batch的生成函数待进一步研究
    def generate(self, start, len):
        #   记序列长度为 L
        #   词汇表大小为 V
        #   隐藏层维数为 H
        #   词嵌入维度为 E
        #   start 为 scalar

        # output = [1]
        out = [start]
        # start = [1, 1, 1]
        if self.device is not None:
            start = torch.tensor(start).unsqueeze(0).unsqueeze(0).to(self.device)
            h = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
            c = torch.zeros(self.num_layers, 1, self.hidden_size).to(self.device)
        else:
            start = torch.tensor(start).unsqueeze(0).unsqueeze(0)
            h = torch.zeros(self.num_layers, 1, self.hidden_size)
            c = torch.zeros(self.num_layers, 1, self.hidden_size)

        # prev_output = [1, 1, E]
        prev_output = self.embedding(start)

        for t in range(len - 1):
            # output = [1, 1, H]
            output, (h, c) = self.lstm(prev_output, (h, c))
            # output = [1, 1, H]，但是是按行的概率分布
            output = F.softmax(self.fc(output), dim=2)
            # prediction = [1, 1, ]
            prediction = torch.argmax(output, 2)
            out.append(prediction.item())
            if prediction.item() == self.vocab_size - 1:
                break

        return out

    def calculate_perplexity(self, x):
        out = self.forward(x)
        return torch.prod(out) ** (- 1 / out.numel())
