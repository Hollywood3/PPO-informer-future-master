import torch
import torch.nn as nn
from model.informer import ProbAttention, FullAttention, EncoderLayer, AttentionLayer, EncoderStack, DataEmbedding, ConvLayer, Encoder
from model.Multihead_Conv import Multihead_Conv
from model.Multihead_informer import Multihead_informer

import warnings
warnings.filterwarnings("ignore")

class Multihead_LSTM_informer_DataEmbedding(nn.Module):
    def __init__(self, cfg):
        super(Multihead_LSTM_informer_DataEmbedding, self).__init__()
        self.cfg = cfg
        Attn = ProbAttention if cfg.attn == 'prob' else FullAttention
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(
                            Attn(False, cfg.factor, attention_dropout=cfg.dropout,
                                 output_attention=cfg.output_attention),
                            cfg.d_model, cfg.n_heads, mix=False),
                        d_model=cfg.d_model,
                        d_ff=cfg.d_ff,
                        dropout=cfg.dropout,
                        activation=cfg.activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        cfg.d_model, stride=1
                    ) for l in range(el - 1)
                ] if cfg.distil else None,
                norm_layer=torch.nn.LayerNorm(cfg.d_model)
            ) for el in cfg.e_layers]

        self.encoder = EncoderStack(encoders, cfg.inp_lens)
        self.DataEmbedding = DataEmbedding(cfg.in_dim, cfg.out_dim, freq=cfg.freq)

        self.Multihead_Conv = Multihead_Conv(cfg.d_model)
        self.Multihead_informer = Multihead_informer(cfg)
        self.norm_layer = torch.nn.LayerNorm(cfg.d_model)

#         self.modle1 = Multihead_LSTM_informer(cfg, 210, stride=1)

#         self.modle2 = Multihead_LSTM_informer(cfg, 367, stride=2)

#         self.modle3 = Multihead_LSTM_informer(cfg, 275, stride=2)

#         self.modle4 = Multihead_LSTM_informer(cfg, 206, stride=2)

#         self.modle5 = Multihead_LSTM_informer(cfg, 155, stride=1)

#         self.modle6 = Multihead_LSTM_informer(cfg, 270, stride=2)

#         self.modle7 = Multihead_LSTM_informer(cfg, 203, stride=2)

#         self.modle8 = Multihead_LSTM_informer(cfg, 152, stride=2)

    def forward(self, x, x_mark):
        # print(x.shape, '\n')
        x = self.DataEmbedding(x, x_mark)
        # print(x.shape, '\n')
        x = self.Multihead_Conv(x)
        # print(x.shape, '\n')
        # x = self.Multihead_LSTM(x)
        # print(x.shape, '\n')
        # x1 = self.Multihead_informer(x1)
#         y, _ = self.LSTM(x)
#         x = self.norm_layer(x + x1)
        x, att = self.encoder(x)

        # x = self.modle1(x)
        # x = self.modle2(x)
        # x = self.modle3(x)
        # x = self.modle4(x)
        # x = self.modle5(x)
        # x = self.modle6(x)
        # x = self.modle7(x)
        # x = self.modle8(x)
        return x


class CFG:
    #  informer
    attn = 'prob'
    dropout = 0.2
    factor = 5
    output_attention = False
    d_model = 64  # 输入dim
    n_heads = 16
    activation = 'gelu'
    distil = True
    e_layers = [3, 2, 1]
    inp_lens = list(range(len(e_layers)))  # [0,1,2,...] you can customize here 张量维度 3维
    d_ff = 64  # 输出dim 两者相等

    # DataEmbedding
    freq = 't'  # freq为t, 时间维度为5, freq为'h', 时间维度为4
    in_dim = 23  # 输入维度
    out_dim = 64  # 输出属性维度
