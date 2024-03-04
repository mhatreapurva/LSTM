import torch
from torch.autograd import Variable
import torch.nn as nn

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.xh = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
        self.ch = nn.Linear(hidden_size, hidden_size)

        self.xi = nn.Linear(input_size, hidden_size)
        self.hi = nn.Linear(hidden_size, hidden_size)
        self.ci = nn.Linear(hidden_size, hidden_size)

        self.xf = nn.Linear(input_size, hidden_size)
        self.hf = nn.Linear(hidden_size, hidden_size)
        self.cf = nn.Linear(hidden_size, hidden_size)

        self.xo = nn.Linear(input_size, hidden_size)
        self.ho = nn.Linear(hidden_size, hidden_size)
        self.co = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h, c):
        breakpoint()
        x = x.view(-1, self.input_size)
        h = h.view(-1, self.hidden_size)
        c = c.view(-1, self.hidden_size)

        i = torch.sigmoid(self.xi(x) + self.hi(h) + self.ci(c))
        f = torch.sigmoid(self.xf(x) + self.hf(h) + self.cf(c))
        o = torch.sigmoid(self.xo(x) + self.ho(h) + self.co(c))
        g = torch.tanh(self.xh(x) + self.hh(h) + self.ch(c))

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = nn.ModuleList([LSTMCell(input_size, hidden_size) for _ in range(num_layers)])

    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        h, c = [], []
        for i in range(self.num_layers):
            h.append(h0[i])
            c.append(c0[i])

        for t in range(x.size(1)):
            for i in range(self.num_layers):
                h[i], c[i] = self.lstm_cells[i](x[:, t], h[i], c[i])

        return h, c

