# -*-coding:utf-8-*-

r"""
从特征到标签的LSGM模型具体计算类
"""

import torch
from torch import optim, nn
from torch.autograd import Variable

from lsgm.utils import only_lab_p, test_val

from tqdm import tqdm
import os
import sys
import random
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
    def __init__(self, lab_num):
        """
        特征到标签类初始函数
        :param lab_num: 实际标签数
        """
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
        self.lab_num = lab_num

    def train_val(self, pos_enc, pos_enc_optim, pos_dec, pos_dec_optim,
                        neg_enc, neg_enc_optim, neg_dec, neg_dec_optim, train_data, val_data):
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
        def _model_train(enc, enc_optim, dec, dec_optim, fea_var, lab_var):
            """
            模型训练核心部分
            :param enc: 编码器
            :param enc_optim: 编码器的优化器
            :param dec: 解码器
            :param dec_optim: 解码器的优化器
            :param fea_var: 特征链变量
            :param lab_var: 标签变量
            :return: 该批次变量计算的损失
            """
            enc.train()  # 编码器训练模式
            enc_optim.zero_grad()  # 编码器优化器梯度清零
            dec.train()  # 解码器训练模式
            enc_optim.zero_grad()  # 解码器优化器梯度清零

            loss = self._train_batch(enc, dec, fea_var, lab_var)  # 模型前向计算

            enc_optim.step()  # 编码器优化器步进，更新参数
            dec_optim.step()  # 解码器优化器步进，更新参数

            return loss

        print('\n===开始训练&验证===\n')

        if config.log:  # 模型写入日志
            pass

        for epoch in range(config.epochs):
            print('\n[训练轮次 {}]'.format(epoch))
            pos_loss_all = 0  # 一个训练轮次所有批次的总损失
            neg_loss_all = 0  # 一个训练轮次所有批次的总损失

            for idx, batch_data in enumerate(train_data):
                fea_var, pos_lab_var, neg_lab_var = batch2tensor(batch_data)  # 转化得到特征变量，正标签变量，负标签变量

                pos_loss = _model_train(pos_enc, pos_enc_optim, pos_dec, pos_dec_optim, fea_var, pos_lab_var)  # 正标签
                neg_loss = _model_train(neg_enc, neg_enc_optim, neg_dec, neg_dec_optim, fea_var, neg_lab_var)  # 负标签

                pos_loss_all += pos_loss
                neg_loss_all += neg_loss

            print('正标记损失：%5.3f' % (pos_loss_all / (idx + 1)))
            print('负标记损失：%5.3f' % (neg_loss_all / (idx + 1)))

            self.val(pos_enc, pos_dec, neg_enc, neg_dec, val_data)  # 每轮次训练结束后，验证模型

    def _train_batch(self, enc, dec, src_seq, tgt_seq):
        """
        模型批次数据训练部分
        :param enc:
        :param dec:
        :param src_seq:
        :param tgt_seq:
        :return:
        """
        enc_out, enc_state = enc(src_seq, None)  # 编码器编码特征链

        dec_state = (enc_state[0][:config.dec_layer], enc_state[1][:config.dec_layer])  # 解码器初始状态=编码最后时刻的状态

        dec_input = Variable(torch.LongTensor([config.sos])).to(config.device)  # 解码器初始输入为sos序列开始符

        tgt_len = len(tgt_seq)  # 标签变量个数

        dec_all_out = Variable(torch.zeros(tgt_len, config.t_batch_size, dec.lab_num)).to(config.device)  # 收集解码器输出

        for i in range(tgt_len):
            dec_out, dec_state, dec_out_p = dec(dec_input, dec_state, enc_out)  # 解码器解码标签

            dec_all_out[i] = dec_out  # 收集解码器的输出

            use_teacher = True if random.random() < config.teach_ratio else False  # 按比例使用教师强制训练机制

            if use_teacher:  # 使用教师强制训练机制，自己的输出作为下一步的输入
                top_value, top_idx = dec_out.data.topk(1)  # 最大值和索引
                idx = int(top_idx[0][0].cpu().numpy())
                dec_input = Variable(torch.LongTensor([idx])).to(config.device)
            else:  # 不使用教师强制训练机制，实际的标签作为下一步的输入
                dec_input = Variable(torch.LongTensor([tgt_seq[i]])).to(config.device)

        dec_all_out = dec_all_out.transpose(0, 1).squeeze(0)  # 全部的预测标签向量
        loss = self.criterion(dec_all_out, tgt_seq)  # 实际的标签
        loss.backward()  # 损失反向传播

        return loss.item()

    def val(self, pos_enc, pos_dec, neg_enc, neg_dec, val_data):
        """
        特征到标签模型的验证部分
        :param pos_enc: 正编码器
        :param pos_dec: 正解码器
        :param neg_enc: 负编码器
        :param neg_dec: 负解码器
        :param val_data: 验证数据
        :return: 预测的标记和该批次数据的损失
        """
        pre_pos_lab, pre_pos_lab_p = [], []  # 预测的正标签和概率
        pre_neg_lab, pre_neg_lab_p = [], []  # 预测的负标签和概率

        real_pos_lab, real_neg_lab = [], []  # 实际的正标签和负标签

        for idx, batch_data in enumerate(val_data):
            fea_var, pos_lab_var, neg_lab_var = batch2tensor(batch_data)  # 转化得到特征变量，正标签变量和负标签变量
            real_pos_lab.append(pos_lab_var)
            real_neg_lab.append(neg_lab_var)

            pos_lab, pos_lab_p = self._val_batch(pos_enc, pos_dec, fea_var, self.lab_num)  # 解码预测正标签
            pre_pos_lab.append(pos_lab)
            pre_pos_lab_p.append(pos_lab_p)

            neg_lab, neg_lab_p = self._val_batch(neg_enc, neg_dec, fea_var, self.lab_num)  # 解码预测负标签
            pre_neg_lab.append(neg_lab)
            pre_neg_lab_p.append(neg_lab_p)

        pre_pos_lab, pre_pos_lab_p = only_lab_p(pre_pos_lab, pre_pos_lab_p)  # 正标签消歧
        pre_neg_lab, pre_neg_lab_p = only_lab_p(pre_neg_lab, pre_neg_lab_p)  # 负标签消歧

        pos_acc = test_val(real_pos_lab, pre_pos_lab, self.lab_num, 0)  # 测试正模型性能
        neg_acc = test_val(real_neg_lab, pre_neg_lab, self.lab_num, 1)  # 测试负模型性能

        print('正模型的验证性能为 %5.3f' % pos_acc)
        print('负模型的验证性能为 %5.3f' % neg_acc)

    def _val_batch(self, enc, dec, src_seq, lab_num):
        """
        模型批次数据验证部分
        :param enc: 编码器
        :param dec: 解码器
        :param src_seq: 特征链变量
        :param lab_num: 总标签数
        :return:
        """
        enc.eval()  # 解码器验证状态
        dec.eval()  # 解码器验证状态

        enc_out, enc_state = enc(src_seq, None)  # 编码器编码特征链

        dec_state = (enc_state[0][:config.dec_layer], enc_state[1][:config.dec_layer])  # 解码器初始状态=编码器最后状态

        dec_input = Variable(torch.LongTensor([config.sos])).to(config.device)  # 解码器第一个输入为sos开始符

        pre_lab = []  # 预测的标签
        pre_lab_p = []  # 预测的标签的概率

        for i in range(lab_num):
            dec_out, dec_state, dec_out_p = dec(dec_input, dec_state, enc_out)  # 解码器解码预测标签

            top_value, top_idx = dec_out.data.topk(1)  # 解码标签向量中最大的值和位置
            idx = int(top_idx[0][0].cpu().numpy())  # 预测出来的标签序号
            if idx == config.eos:  # 如果预测的是eos结束符
                break  # 结束解码循环
            else:
                dec_input = Variable(torch.LongTensor([idx])).to(config.device)  # 解码器预测的作为下一个输入

                pre_lab.append(idx)  # 解码出来的标签

                top_value, top_idx = dec_out_p.data.topk(1)  # 解码标签概率向量中最大的值和位置
                p = int(top_value[0][0].cpu().numpy())  # 预测出来的标签概率

                pre_lab_p.append(p)  # 对应解码标签的解码概率

        return pre_lab, pre_lab_p

    def infer(self, enc, dec, src_seq):
        """
        特征到标签模型的推理部分
        :param enc: 编码器
        :param dec: 解码器
        :param src_seq: 输入特征序列
        :return: 预测的标记
        """
        pass



