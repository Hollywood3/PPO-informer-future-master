import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")


class Multihead_Conv(nn.Module):
    def __init__(self, dim):
        super(Multihead_Conv, self).__init__()
        self.dim = dim
        self.conv_modles = nn.ModuleList(
            torch.nn.Sequential(nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool1d(kernel_size=3, stride=1, padding=1)) for i in range(self.dim))

    def forward(self, x):
        x_stack = []
        for len_, modle in zip(range(x.shape[2]), self.conv_modles):
            a = modle(x[:, :, len_].unsqueeze(1))
            x_stack.append(a)
        x_stack = torch.cat(x_stack, -2)
        x_stack = x_stack.permute(0, 2, 1)

        return x_stack