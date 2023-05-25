import torch
import torch.nn as nn
from model.informer import ProbAttention, FullAttention, EncoderLayer, AttentionLayer

import warnings
warnings.filterwarnings("ignore")

class Multihead_informer(nn.Module):
    def __init__(self, cfg):
        super(Multihead_informer, self).__init__()
        self.cfg = cfg
        Attn = ProbAttention if cfg.attn == 'prob' else FullAttention
        self.informer = EncoderLayer(AttentionLayer(
            Attn(False, cfg.factor, attention_dropout=cfg.dropout, output_attention=cfg.output_attention),
            1, 1, mix=False),
            d_model=1, d_ff=1, dropout=0, activation='gelu', norm_dims=2)

        self.informer_modle = nn.ModuleList(self.informer for i in range(cfg.d_model))

    def forward(self, x):
        x_stack = []
        for len_, modle in zip(range(x.shape[2]), self.informer_modle):
            # print(x[:, :, len_].unsqueeze(2).shape)
            new_x, attn = modle(x[:, :, len_].unsqueeze(2))
            x_stack.append(new_x)
        # print(x_stack)
        x_stack = torch.cat(x_stack, -1)

        return x_stack