import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_LM(nn.Module):
    def __init__(self, vocab_size, dim_emb, hidden_size, num_layers, limit):
        super(LSTM_LM, self).__init__()
        self.vocab_size = vocab_size
        self.dim_emb = dim_emb
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.limit = limit

        self.embedding = nn.Embedding(vocab_size, dim_emb)
        self.lstm = nn.LSTM(dim_emb, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        self.limit = x.size(1)
        out = torch.ones(x.size(0), self.limit)
        h = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        prev_output = self.embedding(x[:, 0].unsqueeze(1))

        for t in range(self.limit):
            output, (h, c) = self.lstm(prev_output, (h, c))
            output = F.softmax(self.fc(output))
            out[:, t + 1] = output[:, x[:, t + 1].unsqueeze(1)]
            prediction = torch.argmax(output, dim=1)

            # 本身这里我想多加个限制，当打出停止符时停止继续生成序列，但是不太好处理，故暂且搁置

            prev_output = self.embedding(prediction)

        return out

    def generate(self, start, length):
