# -*-coding:utf-8-*-

r"""
数据处理模块
"""

import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree, KDTree

import sys
sys.path.append(os.path.abspath('..'))
from config import Config
config = Config()


class Process:
    """

    """
    def __init__(self, data_type, k):
        """
        数据处理类初始化函数
        :param data_type: 多标记数据类型
        :param k: 特征最近邻个数
        """
        self.data_type = data_type
        self.k = k
        self.ins_num = 0
        self.fea_num = 0  # 特征数
        self.lab_num = 0  # 标签数

    def get_data(self):
        """
        获取多标记特征和标记数据
        :return: 特征集和标签集
        """
        data_path = './data/' + self.data_type + '.csv'  # 数据路径
        data_set = np.loadtxt(open(data_path, 'rb'), delimiter=',', skiprows=0)  # 加载数据

        lab_num = 0  # 标签数量

        if self.data_type == 'yeast':
            lab_num = 14
        elif self.data_type == 'scene':
            lab_num = 6
        elif self.data_type == 'emotions':
            lab_num = 6
        elif self.data_type == 'enron':
            lab_num = 53
        elif self.data_type == 'image':
            lab_num = 5

        fea, lab = data_set[:, :-lab_num], data_set[:, -lab_num:]  # 获取特征和标签数据

        if config.fea_norm == 1:  # 特征数据归一化[0, 1]
            fea = preprocessing.MinMaxScaler().fit_transform(fea)
        elif config.fea_norm == 2:  # 特征数据归一化[-1, 1]
            fea = preprocessing.MaxAbsScaler().fit_transform(fea)
        elif config.fea_norm == 0:  # 特征数据不进行归一化
            fea = fea


        self.ins_num, self.fea_num = fea.shape
        self.lab_num = lab_num

        # TODO 当特征维度过小或者过大，是否考虑维度升维和降维

        return fea, lab

    def sort_lab_order(self, lab, order):
        """
        重新排列标签集
        :param lab:  标签集
        :param order: 标签排列顺序 r(random), d(descend), a(ascend)
        :return: 重新排列后的标签集
        """
        if order == 'r':  # 原始顺序就相当于随机顺序
            return lab

        ins_num, lab_num = lab.shape  # 示例数，标签数
        sum_each_col = (lab.sum(axis=0)).tolist()  # 求和每一列
        idx_col = {}  # 每一列的序号
        for i in range(len(sum_each_col)):  # 给每列打上序号
            idx_col[i] = sum_each_col[i]
        reverse = True if order == 'd' else False  # 降序或者升序排列标签
        new_order = sorted(idx_col, key=lambda x: idx_col[x], reverse=reverse)  # 排列顺序
        lab_new = np.zeros((ins_num, lab_num))  # 新标签集
        for i in range(len(new_order)):  # 形成新的标签集
            lab_new[:, i] = lab[:, new_order[i]]

        return lab_new

    def t_v_i(self, fea, lab, t_ratio=0.9, i_ratio=0.8):
        """
        分离数据为训练，验证和推理
        :param fea: 总的特征集
        :param lab: 总的标签集
        :return: 训练，验证和推理的特征集，标签集
        """
        fea_idx = np.column_stack((fea, [i for i in range(fea.shape[0])]))  # 特征集最后加上序号
        lab_idx = np.column_stack((lab, [i for i in range(lab.shape[0])]))  # 标签集最后加上序号

        # 拆分训练，验证-推理的x和y数据
        t_x, v_i_x, \
        t_y, v_i_y = train_test_split(fea_idx, lab_idx, train_size=t_ratio, test_size=1-t_ratio, shuffle=True)

        # 拆分验证，推理的x和y数据
        v_x, i_x, \
        v_y, i_y = train_test_split(v_i_x, v_i_y, train_size=1-i_ratio, test_size=i_ratio, shuffle=True)

        return t_x, t_y, v_x, v_y, i_x, i_y

    def get_k_ins(self, t_fea, v_fea, i_fea):
        """
        将训练的特征集构建为查询树，查询每个示例的k个最近邻
        :param t_fea: 训练的特征集
        :param v_fea: 验证的特征集
        :param i_fea: 测试的特征集
        :return: 对各个特征集查询的k个最近邻结果
        """
        fea_num = t_fea.shape[1] - 1  # 特征维数

        query_tree = BallTree(t_fea[:, :-1], metric=config.tree_metric)  # 构建训练特征集查询树

        def _get_k_ins(tree, train_fea, fea, k, t=None):
            """
            从查询树中找到自己的k个特征最近邻
            :param tree: 查询树
            :param train_fea: 查询的训练特征集
            :param fea: 特征集
            :param k: 特征最近邻个数
            :param t: 是不是训练特征集
            :return: 每个示例的k个特征最近邻
            """
            k_ins_obj = []  # 每个示例的k个特征最近邻
            for i in fea:
                one = i[:-1].reshape((1, fea_num))  # 当前示例的特征
                _, k_ins = tree.query(X=one, k=k)  # 查询k个特征最近邻
                k_ins = list(reversed(k_ins[0, :]))  # 反转k个特征最近邻，按照和当前最近邻的关系从小到大
                k_ins = [int(train_fea[j][-1]) for j in k_ins]  # 找出当前示例实际的特征编码
                if not t:  # 如果是验证和推理特征集，要加上自己
                    k_ins.append(i[-1])
                k_ins_obj.append(k_ins)  # 当前示例的k个最近邻

            return np.array(k_ins_obj)

        t_k_ins = _get_k_ins(query_tree, t_fea, t_fea, self.k + 1, True)  # 对训练特征集查询k+1个，因为是训练特征集构建的查询树，自己也算
        v_k_ins = _get_k_ins(query_tree, t_fea, v_fea, self.k)  # 对验证特征集查询k个最近邻
        i_k_ins = _get_k_ins(query_tree, t_fea, i_fea, self.k)  # 对推理特征集查询k个最近邻

        return t_k_ins, v_k_ins, i_k_ins

    def get_alpha_beta(self, i_k, lab, k):
        """
        获取推理集中每个示例的k个特征最近邻的标记分布的先验概率（正和负）
        :param i_k: 推理集的k个特征最近邻集
        :param lab: 标签集
        :param k: k个最近邻
        :return: 推理集每个示例的k个特征最近邻的标记分布的先验概率alpha和beta
        """
        i_k = i_k[:, :-1]  # 最后一列也就是本示例对应的标签要排除
        alpha, beta = 0, 0  # 初始先验概率
        ins_num = i_k.shape[0]  # 推理集示例数
        lab_num = lab.shape[1]  # 标签数
        alpha = np.zeros((ins_num, lab_num))  # alpha先验概率
        beta = np.zeros((ins_num, lab_num))  # beta先验概率
        for idx, i in enumerate(i_k):
            k_lab = 0  # k个最近邻的标签集
            for j in i:
                k_lab += lab[int(j)]  # 累加每一个最近邻示例的标签集
            alpha[idx] = k_lab / k  # 得到alpha先验概率
            beta[idx] = 1 - k_lab / k  # 得到beta先验概率

        for i in range(ins_num):
            for j in range(lab_num):
                alpha[i][j] = round(alpha[i][j], 3)  # 保留三维小数
                beta[i][j] = round(beta[i][j], 3)

        return alpha, beta

    def build_fea_chain(self, chain_type, train_k, val_k, infer_k):
        """
        构建训练，验证和测试的特征链
            特征链类型
                mix 按照和当前示例的相似度由小到大排列，两两之间插入本示例
                last 按照和当前示例的相似度由小到大排列，最后放上本示例
        :param chain_type: 特征链类型
        :param train_k: 训练集的k个特征最近邻
        :param val_k: 验证集的k个特征最近邻
        :param infer_k: 推理集的k个特征最近邻
        :return: 训练，验证和推理的特征链
        """
        def __build_fea_chain(k_ins):
            """
            构建特征链
            :param k: k个特征最近邻
            :return: 2k长度的特征链
            """
            all_chain = []  # 全体特征链
            for i in k_ins:
                own = i[-1]  # 本示例
                chain = []  # 特征链
                for j in range(k_ins.shape[1] - 1):  # 交替构建最近邻的本示例
                    chain.append(int(i[j]))
                    chain.append(int(own))
                all_chain.append(chain)

            return all_chain

        if  chain_type == 'last':  # 最后放置种类型的特征链
            return train_k, val_k, infer_k

        if chain_type == 'mix':  # 混合类型的特征链
            train_chain = __build_fea_chain(train_k)  # 训练特征链
            val_chain = __build_fea_chain(val_k)  # 验证特征链
            infer_chain = __build_fea_chain(infer_k)  # 推理特征链

            return train_chain, val_chain, infer_chain

    def merge_chain_lab(self, train_chain, val_chain, infer_chain, lab):
        """
        合并特征链和正负标记
        :param train_chain: 训练特征链
        :param val_chain: 验证特征链
        :param infer_chain: 推理特征链
        :param lab: 标签集
        :return: 合并后的训练特征链-标签，验证特征链-标签，推理特征链-标签
        """
        def __merge_chain_lab(chain, lab):
            """
            合并特征链和正负标签
            :param chain: 特征链
            :param lab: 标签集
            :return: 特征链-正负标签
            """
            chain_lab = []  # 特征链-标签
            for i in chain:
                own = i[-1]  # 当前示例
                labels = lab[own]
                pos_lab = []  # 正标签
                neg_lab = []  # 负标签
                for idx, j in enumerate(labels):
                    pos_lab.append(idx + 2) if j > 0 else neg_lab.append(idx + 2)  # 区分正负标记，+2因为有SOS和EOS符
                pos_lab.append(config.eos)  # 正标签序列加上终止符
                neg_lab.append(config.eos)  # 负标签序列加上终止符
                chain_lab.append([str(i), str(pos_lab), str(neg_lab)])  # 合并特征链和标签

            return chain_lab

        t_chain_lab = __merge_chain_lab(train_chain, lab)  # 训练特征链-标签
        v_chain_lab = __merge_chain_lab(val_chain, lab)  # 验证特征链-标签
        i_chain_lab = __merge_chain_lab(infer_chain, lab)  # 推理特征链-标签

        return t_chain_lab, v_chain_lab, i_chain_lab










