# -*-coding:utf-8-*-

r"""
标签序列解码器
"""

import torch
from torch import nn

from lsgm.attention import Attn

import os
import sys
sys.path.append(os.path.abspath('..'))
from config import Config
config = Config()


class LabelSequenceDecoder(nn.Module):
    """
    标签序列解码器
    """
    def __init__(self, lab_num, lab_dim, lab_emb=None, attn='dot'):
        """
        标签序列解码器初始化
        :param lab_num: 标签数量
        :param lab_dim: 标签向量维度
        :param lab_emb: 预训练的标签向量
        :param attn: attention计算方式
        """
        super(LabelSequenceDecoder, self).__init__()
        self.hidden_dim = lab_dim
        self.lab_num = lab_num

        self.lab_emb = nn.Embedding(lab_num, lab_dim)  # 标签向量层

        self.dropout = nn.Dropout(config.dec_dropout)  # 解码器dropout层

        self.lstm = nn.LSTM(input_size=lab_dim,
                            hidden_size=lab_dim,
                            num_layers=config.dec_layer,
                            dropout=config.dec_dropout,
                            batch_first=True)

        # TODO layer norm

        self.concat = nn.Linear(lab_dim * 2, lab_dim)  # attention级联层
        nn.init.xavier_normal_(self.concat.weight)

        self.out = nn.Linear(lab_dim, lab_num, bias=False)  # 输出转化维度层
        nn.init.xavier_normal_(self.out.weight)

        self.attn = Attn(attn, lab_dim)  # attention计算层

        self.softmax = nn.Softmax(dim=1)  # 输出softmax层

    def forward(self, lab_seq, state, enc_outs):
        """
        编码器前向计算层
        :param lab_seq: 上一个标记
        :param enc_outs: 编码器全部的输出
        :return: 解码输出标记，新LSTM状态，各个标签的预测概率
        """
        batch_size = lab_seq.shape[0]  # 批次大小

        lab_emb = self.lab_emb(lab_seq)  # 获取标签向量
        lab_emb = self.dropout(lab_emb)  # 做dropout
        lab_emb = lab_emb.view(1, batch_size, self.hidden_dim)  # 变换为3维，便于送入LSTM中计算

        out, state = self.lstm(lab_emb, state)  # 将标签向量送入LSTM中计算

        attn_w = self.attn(out, enc_outs.transpose(0, 1))  # attention计算
        context = attn_w.bmm(enc_outs)
        concat_input = torch.cat((out.squeeze(0), context.squeeze(1)), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        out = self.out(concat_output)  # 对输出做维度转化，使用交叉熵，那就不需要做softmax

        out_p = self.softmax(out)

        return out, state, out_p



