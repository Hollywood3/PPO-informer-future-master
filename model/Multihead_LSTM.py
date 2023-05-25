import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

class Multihead_LSTM(nn.Module):
    def __init__(self, dim, dropout):
        super(Multihead_LSTM, self).__init__()
        self.dim = dim
        self.lstm_modles = nn.ModuleList(
            [nn.LSTM(input_size=1, hidden_size=1, num_layers=2, batch_first=True, dropout=dropout) for i in
             range(self.dim)])
        self.norm_layer = torch.nn.LayerNorm(self.dim)

    def forward(self, x):
        x_stack = []
        for len_, modle in zip(range(self.dim), self.lstm_modles):
            a = modle(x[:, :, len_].unsqueeze(2))
            x_stack.append(a[0])

        x_stack = torch.cat(x_stack, -1)
        x = self.norm_layer(x + x_stack)
        return x