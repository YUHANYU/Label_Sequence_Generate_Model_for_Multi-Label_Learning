# -*-coding:utf-8-*-

r"""
特征链编码器
"""

import torch
from torch import nn

import os
import sys
sys.path.append(os.path.abspath('..'))
from config import Config
config = Config()

class FeatureChainEncoder(nn.Module):
    """
    特征链编码器
    """
    def __init__(self, ins_num, fea_dim, features):
        """
        初始化特征邻编码器
        :param ins_num: 示例数
        :param fea_dim: 特征维度
        :param features: 全体特征集
        """
        super(FeatureChainEncoder, self).__init__()
        self.hidden_dim = fea_dim  # 隐藏层的维度

        self.fea_emb = nn.Embedding(ins_num, fea_dim)  # 特征向量层
        feature = torch.from_numpy(features).float()
        self.fea_emb.from_pretrained(torch.FloatTensor(feature), freeze=True)  # 加载全体特征向量。并不更新

        self.lstm = nn.LSTM(input_size=fea_dim,
                            hidden_size=fea_dim,
                            num_layers=config.enc_layer,
                            dropout=config.enc_dropout,
                            bidirectional=config.enc_bi,
                            batch_first=True)
        # TODO LSTM初始化权重

        # TODO 层规范化

    def forward(self, input_chain, state):
        """
        编码器前向计算
        :param input_seq: 输入特征链
        :param state: LSTM状态
        :return: 编码特征链后的结果
        """
        chain_emb = self.fea_emb(input_chain).unsqueeze(0).to(config.device)  # 特征链对应的特征向量

        # TODO state的初始化
        out, state = self.lstm(chain_emb, state)  # lstm编码计算

        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]  # 双向相加

        return out, state



