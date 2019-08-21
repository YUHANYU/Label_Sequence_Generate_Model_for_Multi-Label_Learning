# -*-coding:utf-8-*-

r"""
多标记学习测试指标程序
"""

import numpy as np

def find(instance, label1, label2):
    index1 = []
    index2 = []
    for i in range(instance.shape[0]):
        if instance[i] == label1:
            index1.append(i)
        if instance[i] == label2:
            index2.append(i)
    return index1, index2


def findmax(outputs):
    Max = -float("inf")
    index = 0
    for i in range(outputs.shape[0]):
        if outputs[i] > Max:
            Max = outputs[i]
            index = i
    return Max, index


def sort(x):
    temp = np.array(x)
    length = temp.shape[0]
    index = []
    sortX = []
    for i in range(length):
        Min = float("inf")
        Min_j = i
        for j in range(length):
            if temp[j] < Min:
                Min = temp[j]
                Min_j = j
        sortX.append(Min)
        index.append(Min_j)
        temp[Min_j] = 9999999
        # temp[Min_j] = float("inf")
    return temp, index


def findIndex(a, b):
    for i in range(len(b)):
        if a == b[i]:
            return i


def avgprec(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)
            labels_index.append(index1)
            not_labels_index.append(index2)

    aveprec = 0
    for i in range(instance_num):
        tempvalue, index = sort(temp_outputs[i])
        indicator = np.zeros((class_num,))
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            indicator[loc] = 1
        summary = 0
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            # print(loc)
            summary = summary + sum(indicator[loc:class_num]) / (class_num - loc);
        aveprec = aveprec + summary / labels_size[i]
    return aveprec / test_data_num


def Coverage(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        labels_size.append(sum(test_target[i] == 1))
        index1, index2 = find(test_target[i], 1, 0)
        labels_index.append(index1)
        not_labels_index.append(index2)

    cover = 0
    for i in range(test_data_num):
        tempvalue, index = sort(outputs[i])
        temp_min = class_num + 1
        for j in range(labels_size[i]):
            loc = findIndex(labels_index[i][j], index)
            if loc < temp_min:
                temp_min = loc
        cover = cover + (class_num - temp_min)
    return (cover / test_data_num - 1) / class_num


def HammingLoss(predict_labels, test_target):
    labels_num = predict_labels.shape[1]
    test_data_num = predict_labels.shape[0]
    hammingLoss = 0
    for i in range(test_data_num):
        notEqualNum = 0
        for j in range(labels_num):
            if predict_labels[i][j] != test_target[i][j]:
                notEqualNum = notEqualNum + 1
        hammingLoss = hammingLoss + notEqualNum / labels_num
    hammingLoss = hammingLoss / test_data_num
    return hammingLoss


def OneError(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    num = 0
    one_error = 0
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            Max, index = findmax(outputs[i])
            num = num + 1
            if test_target[i][index] != 1:
                one_error = one_error + 1
    return one_error / num


def rloss(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    temp_outputs = []
    temp_test_target = []
    instance_num = 0
    labels_index = []
    not_labels_index = []
    labels_size = []
    for i in range(test_data_num):
        if sum(test_target[i]) != class_num and sum(test_target[i]) != 0:
            instance_num = instance_num + 1
            temp_outputs.append(outputs[i])
            temp_test_target.append(test_target[i])
            labels_size.append(sum(test_target[i] == 1))
            index1, index2 = find(test_target[i], 1, 0)
            labels_index.append(index1)
            not_labels_index.append(index2)

    rankloss = 0
    for i in range(instance_num):
        m = labels_size[i]
        n = class_num - m
        temp = 0
        for j in range(m):
            for k in range(n):
                if temp_outputs[i][labels_index[i][j]] < temp_outputs[i][not_labels_index[i][k]]:
                    temp = temp + 1
        rankloss = rankloss + temp / (m * n)

    rankloss = rankloss / instance_num
    return rankloss


def SubsetAccuracy(predict_labels, test_target):
    test_data_num = predict_labels.shape[0]
    class_num = predict_labels.shape[1]
    correct_num = 0
    for i in range(test_data_num):
        for j in range(class_num):
            if predict_labels[i][j] != test_target[i][j]:
                break
        if j == class_num - 1:
            correct_num = correct_num + 1

    return correct_num / test_data_num


def MacroAveragingAUC(outputs, test_target):
    test_data_num = outputs.shape[0]
    class_num = outputs.shape[1]
    P = []
    N = []
    labels_size = []
    not_labels_size = []
    AUC = 0
    for i in range(class_num):
        P.append([])
        N.append([])

    for i in range(test_data_num):  # 得到Pk和Nk
        for j in range(class_num):
            if test_target[i][j] == 1:
                P[j].append(i)
            else:
                N[j].append(i)

    for i in range(class_num):
        labels_size.append(len(P[i]))
        not_labels_size.append(len(N[i]))

    for i in range(class_num):
        auc = 0
        for j in range(labels_size[i]):
            for k in range(not_labels_size[i]):
                pos = outputs[P[i][j]][i]
                neg = outputs[N[i][k]][i]
                if pos > neg:
                    auc = auc + 1
        print(AUC, auc, labels_size[i], not_labels_size[i])
        AUC = AUC + auc / (labels_size[i] * not_labels_size[i])
    return AUC / class_num


def Performance(predict_labels, test_target):
    data_num = predict_labels.shape[0]
    tempPre = np.transpose(np.copy(predict_labels))
    tempTar = np.transpose(np.copy(test_target))
    tempTar[tempTar == 0] = -1
    com = sum(tempPre == tempTar)
    tempTar[tempTar == -1] = 0
    PreLab = sum(tempPre)
    TarLab = sum(tempTar)
    I = 0
    for i in range(data_num):
        if TarLab[i] == 0:
            I += 1
        else:
            if PreLab[i] == 0:
                I += 0
            else:
                I += com[i] / PreLab[i]
    return I / data_num


def mll_measures(y_pre, y_tgt):
    """
    衡量模型的多标记学习指标
    :param y_pre: 预测标签集
    :param y_tgt: 目标标签集
    :return: 衡量指标
    """
    hamming_loss = round(HammingLoss(y_tgt, y_pre), 3)
    coverage = round(Coverage(y_pre, y_tgt), 3)
    one_error = round(OneError(y_pre, y_tgt), 3)
    rank_loss = round(rloss(y_pre, y_tgt), 3)

    average_p = round(avgprec(y_pre, y_tgt), 3)
    subset_acc = round(SubsetAccuracy(y_pre, y_tgt), 3)

    print('模型推理结果！')

    print('越大越好'
          '| 子集准确率{}'
          '| 平均精确度{}'.format(subset_acc, average_p))

    print('越小越好'
          '| 汉明损失{}'
          '| 唯一错误率{}'
          '| 平均度{}'
          '| 排名损失{}\n'.format(hamming_loss, coverage, one_error, rank_loss))

    return [hamming_loss, coverage, one_error, rank_loss, average_p, subset_acc]



