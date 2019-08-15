# -*-coding:utf-8-*-

r"""
从特征到标签的LSGM模型具体计算类
"""

import torch
from torch import optim, nn
from torch.autograd import Variable

from encoder import FeatureChainEncoder
from decoder import LabelSequenceDecoder

from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath('..'))
from config import Config
config = Config()


def batch2tensor(batch_data):
    """
    转化batch数据为tensor，分为特征链，正标标签，负标签
    :param batch_data: 批次数据
    :return: 特征链tensor，正标签tensor，负标签tensor
    """
    def str2list(target_str):
        target_str = target_str.lstrip('[').rstrip(']').replace(' ', '').split(',')
        target_list = [int(target_str[i]) for i in range(len(target_str))]
        return target_list

    fea = str2list(batch_data[0][0])  # 特征链
    pos_lab = str2list(batch_data[1][0])  # 正标签
    neg_lab = str2list(batch_data[2][0])  # 负标签

    fea_var = Variable(torch.LongTensor(fea)).to(config.device)  # 转化为tensor
    pos_lab_var = Variable(torch.LongTensor(pos_lab)).to(config.device)  # 转化为tensor
    neg_lab_var = Variable(torch.LongTensor(neg_lab)).to(config.device)  # 转化为tensor

    return fea_var, pos_lab_var, neg_lab_var


class Feature2Label:
    """
    特征序列到标签序列计算类
    """
    def __init__(self):
        """
        特征到标签类初始函数
        """
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def train_val(self, enc, enc_optim, dec, dec_optim, train_data, val_data):
        """
        特征到标签模型的的训练部分
        :param enc: 编码器
        :param enc_optim: 编码器的优化器
        :param dec: 解码器
        :param dec_optim: 解码器的优化器
        :param train_data: 训练数据
        :param val_data: 验证数据
        :return:
        """
        print('\n===开始训练&验证===\n')

        if config.log:  # 模型写入日志
            pass

        for epoch in range(config.epochs):
            print('[训练轮次 {}]'.format(epoch))

            for batch_data in tqdm(train_data, desc='Training...', leave=False):
                enc_optim.zero_grad()  # 编码器优化器梯度清零
                dec_optim.zero_grad()  # 解码器优化器梯度清零

                enc.train()  # 编码器训练模式
                dec.train()  # 解码器训练模式

                fea_var, pos_lab_var, neg_lab_var = batch2tensor(batch_data)  # 转化得到特征变量，正标签变量，负标签变量

    def _train_batch(self, enc, dec, src_seq, tgt_seq):
        pass

    def val(self, enc, dec, src_seq):
        """
        特征到标签模型的验证部分
        :param enc: 编码器
        :param dec: 解码器
        :param src_seq: 输入特征序列
        :return: 预测的标记和该批次数据的损失
        """
        pass

    def infer(self, enc, dec, src_seq):
        """
        特征到标签模型的推理部分
        :param enc: 编码器
        :param dec: 解码器
        :param src_seq: 输入特征序列
        :return: 预测的标记
        """
        pass



