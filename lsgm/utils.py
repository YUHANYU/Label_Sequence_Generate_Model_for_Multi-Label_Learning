# -*-coding:utf-8-*-

r"""
LSGM模型工具程序
"""

from torch.utils.tensorboard import SummaryWriter
import numpy as np

from lsgm.evaluate import avgprec

writer = SummaryWriter()

for i in range(1000):
    writer.add_scalar('loss/train', np.random.random(), i)


def only_lab_p(lab, p):
    """
    对于验证或推理生成的标签和对应的概率进行消歧
    :param lab: 标签集
    :param p: 标签对应的概率集
    :return: 消歧后的标签和对应的概率
    """
    new_lab, new_lab_p = [], []  # 新生成的标签和对应的概率
    for idx_1, i in enumerate(lab):  # 循环标签集中每个list
        lab_p = {}  # 组成标签-概率字典
        for idx_2, j in enumerate(i):  # 循环list的每个元素
            if j not in lab_p.keys():  # 如果没有这表标签
                lab_p[j] = p[idx_1][idx_2]  # 形成标签-概率的键值
            else:  # 如果有这个标签
                lab_p[j] = max(p[idx_1][idx_2], lab_p[j])  # 取最大的标签概率

        temp_lab, temp_lab_p = [], []  # 获取标签-概率字典中的标签和概率
        for key, value in lab_p.items():
            temp_lab.append(key)  # 获取标签
            temp_lab_p.append(value)  # 获取概率

        new_lab.append(temp_lab)
        new_lab_p.append(temp_lab_p)

    return new_lab, new_lab_p

def test_val(real_lab, pre_lab, lab_num, lab_mark):
    """
    测试验证过程模型的性能，以多标记学习的平均准确率为标准
    :param real_lab: 实际的标签集
    :param pre_lab: 预测的标签集
    :param lab_num: 标签数
    :param lab_mark: 标签标记，如果是正模型，则是0；负模型，则是1
    :return: 平均准确率
    """
    ins_num = len(real_lab)  # 验证示例数

    # 还原实际标签对应的0-1位置标签
    real_lab = [[int(j.cpu().numpy()) for j in i] for i in real_lab]  # tensor2list
    real_lab = [[j - 2 for j in i[:-1]] for i in real_lab]  # -2是消除开始符合结束符，还原成原来的标签位序
    real_lab_0_1 = [[int(lab_mark) for _ in range(lab_num)] for _ in range(ins_num)]  # 实际的0-1标签
    for idx, i in enumerate(real_lab):
        for j in i:
            real_lab_0_1[idx][j] = int(abs(lab_mark - 1))  # 找到标签的位序，对应位序的位置上，标签置为lab_mark的相反标记

    pre_lab = [[j - 2 for j in i] for i in pre_lab]  # -2是为了消除开始符合结束符，还原成实际的标签位序
    pre_lab_0_1 = [[None for _ in range(lab_num)] for _ in range(ins_num)]  # 预测的0-1标签
    for idx, i in enumerate(pre_lab):
        for j in i:
            pre_lab_0_1[idx][j] = int(abs(lab_mark - 1))  # 找到预测标签的位序，对应位序的位置上，标签置为lab_mark的相反标记

    pre_lab_0_1 = [[int(lab_mark) if j == None else int(j) for j in i] for i in pre_lab_0_1]  # 为了进行测评，需要转化None标记为lab_mark

    average_acc = avgprec(np.array(pre_lab_0_1), real_lab_0_1)  # 平均准确率

    # TODO 对应的概率也要这样操作

    return average_acc









