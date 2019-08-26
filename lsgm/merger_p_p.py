# -*-coding:utf-8-*-

r"""
融合标签的先验概率和生成概率
"""

from .utils import real_2_real, pre_2_pre, mll_measures

import numpy as np

from config import Config
config = Config()


def merge_p_p(alpha, beta, real_lab_pos, real_lab_neg, pos_lab, pos_lab_p, neg_lab, neg_lab_p, ins_num, lab_num, f):
    """
    根据标签的先验概率和生成概率，基于已初步预测出来的标签和概率，根据概率进行融合
    :param real_lab_pos: 正编码-解码的真实标签集
    :param real_lab_neg: 负编码-解码的真实标签集
    :param alpha: 正先验概率
    :param beta: 负先验概率
    :param real_lab: 真实的标签集
    :param pos_lab: 预测的正标签集
    :param pos_lab_p: 对应正标签集的生成概率集
    :param neg_lab: 预测的负标签集
    :param neg_lab_p: 对应负标签集的生成概率集
    :param ins_num: 示例数
    :param lab_num: 标签数
    :return: 融合概率优化后的标签集
    """
    for i in range(ins_num):
        for j in range(lab_num):
            assert real_lab_pos[i][j] == real_lab_neg[i][j]
            '正编码-解码器的真实标签和负编码-解码器的真实标签不一致！！！'

    merge_lab = [[None for _ in range(lab_num)] for _ in range(ins_num)]

    for i in range(ins_num):
        for j in range(lab_num):
            p, q = 0, 0
            if pos_lab[i][j] == None:
                q += 0.6
            else:
                p = pos_lab_p[i][j]

            if neg_lab[i][j] == None:
                p += 0.4
            else:
                q = neg_lab_p[i][j]

            if p * alpha[i][j] * config.alpha > q * beta[i][j] * config.beta:
                merge_lab[i][j] = 1
            else:
                merge_lab[i][j] = 0

    print('融合正负模型推理结果！')
    f.write('融合正负模型推理结果！\n')
    results = mll_measures(np.array(merge_lab), real_lab_pos, f)
    f.write('\n\n')

    return results
