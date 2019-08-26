# -*-coding:utf-8-*-

r"""
从特征到标签的LSGM模型具体计算类
"""

import torch
from torch import optim, nn
from torch.autograd import Variable

from lsgm.utils import only_lab_p, test_val, test_infer
from lsgm.merger_p_p import merge_p_p
from lsgm.encoder import FeatureChainEncoder
from lsgm.decoder import LabelSequenceDecoder

import numpy as np
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
    def __init__(self, ins_num, lab_num, alpha, beta):
        """
        特征到标签类初始函数
        :param lab_num: 实际标签数
        """
        self.criterion = nn.CrossEntropyLoss()  # 采用交叉熵损失函数计算预测和目标
        self.alpha = alpha  # 标签正先验概率
        self.beta = beta  # 负先验概率
        self.ins_num = ins_num
        self.lab_num = lab_num
        self.pos_val_acc = [0]  # 正编码-解码器验证所有批次的准确率
        self.neg_val_acc = [0]  # 负编码-解码器验证所有批次的准确率
        self.pos_save_checkpoint = 0  # 正编码-解码器最佳保存点次数
        self.neg_save_checkpoint = 0  # 负编码-解码器最佳保存点次数


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

        if config.log:  # 模型写入训练日志
            write_path = os.path.abspath('.') + '/results/'  # 日志写入路径
            pos_train_log = open(write_path + 'train_pos_log.log', 'w', encoding='utf-8')  # 正训练日志
            pos_train_log.write(
                '轮次：{epoch:3.0f},'
                '损失：{loss:5.3f}\n\n'
                .format(epoch=0, loss=0))

            neg_train_log = open(write_path + 'train_neg_log.log', 'w', encoding='utf-8')  # 正训练日志
            neg_train_log.write(
                '轮次：{epoch:3.0f},'
                '损失：{loss:5.3f}\n\n'
                .format(epoch=0, loss=0))

        for epoch in range(config.epochs):
            print('[训练轮次 {}]'.format(epoch))

            pos_loss_all = 0  # 正：一个训练轮次所有批次的总损失
            neg_loss_all = 0  # 负：一个训练轮次所有批次的总损失

            for batch_data in tqdm(train_data, desc='Training', leave=False):  # 迭代批次数据，训练模型
                fea_var, pos_lab_var, neg_lab_var = batch2tensor(batch_data)  # 转化得到特征变量，正标签变量，负标签变量

                pos_loss = _model_train(pos_enc, pos_enc_optim, pos_dec, pos_dec_optim, fea_var, pos_lab_var)  # 正损失
                neg_loss = _model_train(neg_enc, neg_enc_optim, neg_dec, neg_dec_optim, fea_var, neg_lab_var)  # 负损失

                if config.log:  # 训练日志
                    pos_train_log.write(
                        '轮次：{epoch:3.0f},'
                        '损失：{loss:5.3f}\n'
                        .format(epoch=epoch, loss=pos_loss))

                    neg_train_log.write(
                        '轮次：{epoch:3.0f},'
                        '损失：{loss:5.3f}\n'
                        .format(epoch=epoch, loss=neg_loss))

                pos_loss_all += pos_loss  # 收集每一批次正编码-解码的损失
                neg_loss_all += neg_loss  # 收集每一批次负编码-解码的损失

            print('正标记损失：%5.3f' % pos_loss_all)  # 显示所有正编码-解码批次数据的总损失
            print('负标记损失：%5.3f' % neg_loss_all)  # 显示所有负编码-解码批次数据的总损失

            self.val(pos_enc, pos_dec, neg_enc, neg_dec, val_data)  # 每轮次训练结束后，验证模型

        pos_train_log.close()  # 关闭文件
        neg_train_log.close()  # 关闭文件

    def _train_batch(self, enc, dec, src_seq, tgt_seq):
        """
        模型批次数据训练部分
        :param enc: 编码器
        :param dec: 解码器
        :param src_seq: 特征链输入序列
        :param tgt_seq: 标签目标序列
        :return: 该批次的损失
        """
        enc_out, enc_state = enc(src_seq, None)  # 编码器编码特征链

        dec_state = (enc_state[0][:config.dec_layer], enc_state[1][:config.dec_layer])  # 解码器初始状态=编码最后时刻的状态

        if config.gli:  # 如果使用全局标签信息
            dec_input = [config.sos] # 标签输入将会存储每一个已经预测出来的标签信息
        else:
            dec_input = Variable(torch.LongTensor([config.sos])).to(config.device)  # 解码器初始输入为sos序列开始符

        tgt_len = len(tgt_seq)  # 标签变量个数

        dec_all_out = Variable(torch.zeros(tgt_len, config.t_batch_size, dec.lab_num)).to(config.device)  # 收集解码器输出

        for i in range(tgt_len):  # 在目标标签个数内预测解码，训练摩西
            dec_out, dec_state, dec_out_p = dec(dec_input, dec_state, enc_out)  # 解码器解码标签

            dec_all_out[i] = dec_out  # 收集解码器的输出

            use_teacher = True if random.random() < config.teach_ratio else False  # 按比例使用教师强制训练机制

            if use_teacher:  # 使用教师强制训练机制，自己的输出作为下一步的输入
                top_value, top_idx = dec_out.data.topk(1)  # 最大值和索引
                idx = int(top_idx[0][0].cpu().numpy())
                if config.gli:  # 如果使用全局标签信息
                    dec_input.append(idx)
                else:  # 不使用全局标签信息
                    dec_input = Variable(torch.LongTensor([idx])).to(config.device)
            else:  # 不使用教师强制训练机制，实际的标签作为下一步的输入
                if config.gli:  # 如果使用全局标签信息
                    dec_input.append(int(tgt_seq[i].cpu().numpy()))  # FIXME 这里要把tensor进行转化
                else:
                    dec_input = Variable(torch.LongTensor([tgt_seq[i]])).to(config.device)

        dec_all_out = dec_all_out.transpose(0, 1).squeeze(0)  # 全部的预测标签向量
        loss = self.criterion(dec_all_out, tgt_seq)  # 计算预测的标签向量和实际的标签序号之间的损失
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
        :return: 验证模型性能，保存模型最佳点
        """
        pre_pos_lab, pre_pos_lab_p = [], []  # 预测的正标签和概率
        pre_neg_lab, pre_neg_lab_p = [], []  # 预测的负标签和概率

        real_pos_lab, real_neg_lab = [], []  # 实际的正标签和负标签

        with torch.no_grad():  # 无梯度，不跟新模参数
            for batch_data in tqdm(val_data, desc='Validating', leave=False):  # 迭代批次数据，验证模型
                fea_var, pos_lab_var, neg_lab_var = batch2tensor(batch_data)  # 转化得到特征变量，正标签变量和负标签变量
                real_pos_lab.append(pos_lab_var)  # 收集每一批次实际的正标签
                real_neg_lab.append(neg_lab_var)  # 收集每一批次实际的负标签

                pos_lab, pos_lab_p = self._val_batch(pos_enc, pos_dec, fea_var, self.lab_num)  # 解码预测正标签
                pre_pos_lab.append(pos_lab)  # 收集每一批次预测的正标签
                pre_pos_lab_p.append(pos_lab_p)  # 对应的概率

                neg_lab, neg_lab_p = self._val_batch(neg_enc, neg_dec, fea_var, self.lab_num)  # 解码预测负标签
                pre_neg_lab.append(neg_lab)  # 收集每一批次预测的负标签
                pre_neg_lab_p.append(neg_lab_p)  #对应的概率

        pre_pos_lab, pre_pos_lab_p = only_lab_p(pre_pos_lab, pre_pos_lab_p)  # 正标签消歧
        pos_acc = test_val(real_pos_lab, pre_pos_lab, self.lab_num, 0)  # 测试正模型平均准确率
        print('正模型的验证平均准确率为 %5.3f' % pos_acc)
        self.pos_val_acc.append(pos_acc)  # 收集本次验证的正编码-解码器准确率
        if pos_acc >= max(self.pos_val_acc):  # 如果当前正准率比以前的都大
            self.pos_save_checkpoint += 1  # 正次数自加1
            pos_enc_state = pos_enc.state_dict()  # 保存正编码器最佳状态
            pos_enc_checkpoint = {  # 正编码器检查点
                'pos_enc': pos_enc_state,
                'setting': config}
            pos_enc_name = os.path.abspath('.') + '/results/pos_enc.chkpt'
            torch.save(pos_enc_checkpoint, pos_enc_name)  # 保存正编码器最佳点

            pos_dec_state = pos_dec.state_dict()  # 保存正解码器最佳状态
            pos_dec_checkpoint = {  # 正解码器检查点
                'pos_dec': pos_dec_state,
                'setting': config}
            pos_dec_name = os.path.abspath('.') + '/results/pos_dec.chkpt'
            torch.save(pos_dec_checkpoint, pos_dec_name)  # 保存正编码器最佳点

            print('已经第{}次更新正编码-解码器模型最佳保存点！'.format(self.pos_save_checkpoint))

        pre_neg_lab, pre_neg_lab_p = only_lab_p(pre_neg_lab, pre_neg_lab_p)  # 负标签消歧
        neg_acc = test_val(real_neg_lab, pre_neg_lab, self.lab_num, 1)  # 测试负模型平均准确率
        print('负模型的验证平均准确率为 %5.3f' % neg_acc)
        self.neg_val_acc.append(neg_acc)  # 收集本次验证的负编码-解码器准确率
        if neg_acc >= max(self.neg_val_acc):  # 如果当前负准率比以前的都大
            self.neg_save_checkpoint += 1  # 负次数自加1
            neg_enc_state = neg_enc.state_dict()  # 保存负编码器最佳状态
            neg_enc_checkpoint = {  # 负编码器检查点
                'neg_enc': neg_enc_state,
                'setting': config}
            neg_enc_name = os.path.abspath('.') + '/results/neg_enc.chkpt'
            torch.save(neg_enc_checkpoint, neg_enc_name)  # 保存负编码器最佳点

            neg_dec_state = neg_dec.state_dict()  # 保存负解码器最佳状态
            neg_dec_checkpoint = {  # 负解码器检查点
                'neg_dec': neg_dec_state,
                'setting': config}
            neg_dec_name = os.path.abspath('.') + '/results/neg_dec.chkpt'
            torch.save(neg_dec_checkpoint, neg_dec_name)  # 保存负编码器最佳点

            print('已经第{}次更新负编码-解码器模型最佳保存点！'.format(self.neg_save_checkpoint))

    def _val_batch(self, enc, dec, src_seq, lab_num):
        """
        模型批次数据解码预测验证部分
        :param enc: 编码器
        :param dec: 解码器
        :param src_seq: 特征链变量
        :param lab_num: 总标签数
        :return: 该批次预测的标签和对应的概率
        """
        enc.eval()  # 解码器验证状态
        dec.eval()  # 解码器验证状态

        enc_out, enc_state = enc(src_seq, None)  # 编码器编码特征链

        dec_state = (enc_state[0][:config.dec_layer], enc_state[1][:config.dec_layer])  # 解码器初始状态=编码器最后状态

        if config.gli:  # 如果使用全局标签信息
            dec_input = [config.sos]
        else:
            dec_input = Variable(torch.LongTensor([config.sos])).to(config.device)  # 解码器第一个输入为sos开始符

        pre_lab = []  # 预测的标签
        pre_lab_p = []  # 预测的标签的概率

        for i in range(lab_num):  # 限定解码标签个数
            dec_out, dec_state, dec_out_p = dec(dec_input, dec_state, enc_out)  # 解码器解码预测标签

            top_value, top_idx = dec_out.data.topk(1)  # 解码标签向量中最大的值和位置
            idx = int(top_idx[0][0].cpu().numpy())  # 预测出来的标签序号
            if idx == config.eos:  # 如果预测的是eos结束符
                break  # 结束解码循环
            else:  # 预测解码出来的不是eos结束符
                if config.gli:  # 如果使用全局标签信息
                    dec_input.append(idx)
                else:
                    dec_input = Variable(torch.LongTensor([idx])).to(config.device)  # 解码器预测的作为下一个输入

                top_value, top_idx = dec_out_p.data.topk(1)  # 解码标签概率向量中最大的值和位置
                p = int(top_value[0][0].cpu().numpy())  # 预测出来的标签概率

                pre_lab.append(idx)  # 收集解码出来的标签
                pre_lab_p.append(p)  # 对应解码标签的解码概率

        return pre_lab, pre_lab_p

    def infer(self, data_obj, infer_data, fea, f):
        """
        特征到标签模型的推理部分
        :param data_obj: 数据对象
        :param infer_data: 推理的数据
        :param fea: 全体特征集
        :return: 预测的标记
        """
        print('\n===推理部分===\n')

        checkpoint_path = os.path.abspath('.') + '/results/'  # 保存点路径

        pos_enc = FeatureChainEncoder(data_obj.ins_num, data_obj.fea_num, fea).to(config.device)  # 正特征编码器
        pos_enc_checkpoint = torch.load(checkpoint_path + 'pos_enc.chkpt', map_location=config.cpu_or_gpu)  # 对应的保存点
        pos_enc.load_state_dict(pos_enc_checkpoint['pos_enc'])  # 加载状态

        pos_dec = LabelSequenceDecoder(data_obj.lab_num + 2, data_obj.fea_num).to(config.device)  # 正标签解码器
        pos_dec_checkpoint = torch.load(checkpoint_path + 'pos_dec.chkpt', map_location=config.cpu_or_gpu)  # 对应的保存点
        pos_dec.load_state_dict(pos_dec_checkpoint['pos_dec'])  # 加载状态

        neg_enc = FeatureChainEncoder(data_obj.ins_num, data_obj.fea_num, fea).to(config.device)  # 正特征编码器
        neg_enc_checkpoint = torch.load(checkpoint_path + 'neg_enc.chkpt', map_location=config.cpu_or_gpu)  # 对应的保存点
        neg_enc.load_state_dict(neg_enc_checkpoint['neg_enc'])  # 加载状态

        neg_dec = LabelSequenceDecoder(data_obj.lab_num + 2, data_obj.fea_num).to(config.device)  # 正标签解码器
        neg_dec_checkpoint = torch.load(checkpoint_path + 'neg_dec.chkpt', map_location=config.cpu_or_gpu)  # 对应的保存点
        neg_dec.load_state_dict(neg_dec_checkpoint['neg_dec'])  # 加载状态

        print('\n===加载预训练模型完成===\n')

        pre_pos_lab = []  # 推理的正标签
        pre_pos_lab_p = []  # 推理的正标签对应的生成概率

        pre_neg_lab = []  # 推理的负标签
        pre_neg_lab_p = []  # 推理的负标签对应的生成概率

        real_pos_lab = []  # 实际的正标签
        real_neg_lab = []  # 实际的负标签

        with torch.no_grad():
            for batch_data in tqdm(infer_data, desc='Inferring', leave=False):
                fea_var, pos_lab_var, neg_lab_var = batch2tensor(batch_data)  # 获取特征链变量，正标记变量，负标记变量
                real_pos_lab.append(pos_lab_var)
                real_neg_lab.append(neg_lab_var)

                pos_lab, pos_lab_p = self._infer_batch(pos_enc, pos_dec, fea_var, self.lab_num)  # 正编码-解码器推理
                pre_pos_lab.append(pos_lab)
                pre_pos_lab_p.append(pos_lab_p)

                neg_lab, neg_lab_p = self._infer_batch(neg_enc, neg_dec, fea_var, self.lab_num)  # 负编码-解码器推理
                pre_neg_lab.append(neg_lab)
                pre_neg_lab_p.append(neg_lab_p)

        pre_pos_lab, pre_pos_lab_p = only_lab_p(pre_pos_lab, pre_pos_lab_p)  # 正标签和对应的概率集消歧
        real_lab_pos, pre_pos_lab, pre_pos_lab_p, pos_result = test_infer(
            real_pos_lab, pre_pos_lab, pre_pos_lab_p, self.lab_num, 0, '正', f)  # 测评模型

        pre_neg_lab, pre_neg_lab_p = only_lab_p(pre_neg_lab, pre_neg_lab_p)  # 负标签消歧
        real_lab_neg, pre_neg_lab, pre_neg_lab_p, neg_result = test_infer(
            real_neg_lab, pre_neg_lab, pre_neg_lab_p, self.lab_num, 1, '负', f)  # 测评模型

        merge_result = merge_p_p(self.alpha, self.beta,  #  融合先验概率和生成概率，再次测评模型
                                 np.array(real_lab_pos), np.array(real_lab_neg),
                                 np.array(pre_pos_lab), np.array(pre_pos_lab_p),
                                 np.array(pre_neg_lab), np.array(pre_neg_lab_p),
                                 self.ins_num, self.lab_num, f)
        f.write('\n\n')

        return pos_result, neg_result, merge_result

    def _infer_batch(self, enc, dec, src_seq, lab_num):
        """
        推理部分批次编码-解码器
        :param enc: 编码器
        :param dec: 解码器
        :param fea: 特征链
        :param lab_num: 标签数
        :return:
        """
        enc.eval()  # 解码器验证状态
        dec.eval()  # 解码器验证状态

        enc_out, enc_state = enc(src_seq, None)  # 编码器编码特征链

        dec_state = (enc_state[0][:config.dec_layer], enc_state[1][:config.dec_layer])  # 解码器初始状态=编码器最后状态

        if config.gli:  # 如果使用全局标签信息
            dec_input = [config.sos]
        else:
            dec_input = Variable(torch.LongTensor([config.sos])).to(config.device)  # 解码器第一个输入为sos开始符

        pre_lab = []  # 预测的标签
        pre_lab_p = []  # 预测的标签的概率

        for i in range(lab_num):  # 限定解码标签个数
            dec_out, dec_state, dec_out_p = dec(dec_input, dec_state, enc_out)  # 解码器解码预测标签

            top_value, top_idx = dec_out.data.topk(1)  # 解码标签向量中最大的值和位置
            idx = int(top_idx[0][0].cpu().numpy())  # 预测出来的标签序号
            if idx == config.eos:  # 如果预测的是eos结束符
                break  # 结束解码循环
            else:  # 预测解码出来的不是eos结束符
                if config.gli:  # 如果使用全局标签信息
                    dec_input.append(idx)
                else:
                    dec_input = Variable(torch.LongTensor([idx])).to(config.device)  # 解码器预测的作为下一个输入

                top_value, top_idx = dec_out_p.data.topk(1)  # 解码标签概率向量中最大的值和位置
                p = round(float(top_value[0][0].cpu().numpy()), 3)  # 预测出来的标签概率

                pre_lab.append(idx)  # 收集解码出来的标签
                pre_lab_p.append(p)  # 对应解码标签的解码概率

        return pre_lab, pre_lab_p

    def _infer_batch_beam_search(self, enc, dec, src_seq, lab_num):
        """
        推理部分编码-解码器，加入beam search 技术
        :param enc:
        :param dec:
        :param src_seq:
        :param lab_num:
        :param beam_size:
        :return:
        """
        enc.eval()
        dec.eval()

        enc_out, enc_state = enc(src_seq, None)  # 编码器编码特征链

        dec_state = (enc_state[0][:config.dec_layer], enc_state[1][:config.dec_layer])  # 解码器初始状态=编码器最后时刻状态

        if config.gli:  # 如果使用全局标签信息
            dec_input = [config.sos]
        else:
            dec_input = Variable(torch.LongTensor([config.sos])).to(config.device)

        pre_lab = []  # 预测出来的标签序列
        pre_lab_b = []  # 对应标签序列的概率

        dec_out, dec_state, dec_out_p = dec(dec_input, dec_state, enc_out)  # 解码器解码sos标签
        top_value, top_idx = dec_out.data.topk(config.beam_size)  # 找出top beam size个下一个token
        idx = int(top_idx.squeeze(0).cpu().numpy())  # beam size个token

        lab_count = 0  # 解码标签数据预测
        while lab_count <= lab_num - 1:  # 最长预测序列长度为标签总数
            count_idx = []
            for j in idx:   # 没一个token都要继续进行解码预测
                if config.gli:  # 如果使用全局标签信息
                    dec_input = dec_input.append(j)
                else:
                    dec_input = Variable(torch.LongTensor([j])).to(config.device)

                dec_out, dec_state, dec_out_p = dec(dec_input, dec_state, enc_out)  # 解码器解码sos标签
                top_value, top_idx = dec_out.data.topk(config.beam_size)  # 找出top beam size个下一个token
                count_idx.append()



        return pre_lab, pre_lab_b

