# -*-coding;utf-8-*-

r"""
LSGM模型的主函数
"""

from torch.utils.data import DataLoader
from torch import optim

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

from data.process import Process
from lsgm.encoder import FeatureChainEncoder
from lsgm.decoder import LabelSequenceDecoder
from lsgm.feature2label import Feature2Label
from lsgm.utils import mean_std

from config import Config
config = Config()


def main(data_type, k):
    print('多标记数据类型{} | 最近邻个数{} | {}使用全局标签信息'.format(data_type, k, '有' if config.gli else '没有'))

    data = Process(data_type, k)  # 数据对象

    fea, lab = data.get_data()  # 获取特征集，标签集

    lab_new = data.sort_lab_order(lab, 'd')  # 降序排列标签集

    fea_idx = np.column_stack((fea, [i for i in range(fea.shape[0])]))  # 特征集最后加上序号
    lab_new_idx = np.column_stack((lab_new, [i for i in range(lab_new.shape[0])]))  # 标签集最后加上序号

    kf = KFold(n_splits=10, shuffle=True)  # 10折交叉训练，验证，推理模型
    pos_results = []  # 正结果
    neg_results = []  # 负结果
    merge_results = []  # 融合结果

    idx = 0  # 训练-验证-推理次数计数
    for train_idx, val_infer_idx in kf.split(fea_idx):  # 9-1切分训练，验证-测试数据
        idx += 1
        print('模型第{}次训练-验证-推理'.format(idx))

        train_fea, train_lab = fea_idx[train_idx], lab_new_idx[train_idx]  # 90份的训练数据
        val_infer_fea, val_infer_lab = fea_idx[val_infer_idx], lab_new_idx[val_infer_idx]  # 10份的验证-推理数据
        val_fea, infer_fea, val_lab, infer_lab = train_test_split(  # 2份的验证数据，8份的推理数据
            val_infer_fea, val_infer_lab, train_size=0.2, test_size=0.8, shuffle=True)

        train_k, val_k, infer_k = data.get_k_ins(train_fea, val_fea, infer_fea)  # 查询训练，验证和测试中每示例的k个最近邻对象

        alpha, beta = data.get_alpha_beta(infer_k, lab_new, k)  # 获得推理集的标签先验概率alpha和beta

        train_chain, val_chain, infer_chain = data.build_fea_chain('mix', train_k, val_k, infer_k)  # 构建训练，验证和推理特征链

        train_chain_lab, val_chain_lab, infer_chain_lab = data.merge_chain_lab(  # 构建训练特&验证&推理的征链-正负标签序列
            train_chain, val_chain, infer_chain, lab_new)

        train_data = DataLoader(train_chain_lab, config.t_batch_size, shuffle=False, drop_last=False)  # 训练数据
        val_data = DataLoader(val_chain_lab, config.v_batch_size, shuffle=False, drop_last=False)  # 验证数据
        infer_data = DataLoader(infer_chain_lab, config.i_batch_size, shuffle=False, drop_last=False)  # 推理数据

        pos_enc = FeatureChainEncoder(data.ins_num, data.fea_num, fea).to(config.device)  # 负编码器
        pos_enc_optim = optim.Adam(pos_enc.parameters(), lr=config.lr_1)  # 负编码器的优化器
        pos_dec = LabelSequenceDecoder(data.lab_num + 2, data.fea_num).to(config.device)  # 负解码器
        pos_dec_optim = optim.Adam(pos_dec.parameters(), lr=config.lr_1)  # 负解码器的优化器

        neg_enc = FeatureChainEncoder(data.ins_num, data.fea_num, fea).to(config.device)  # 负编码器
        neg_enc_optim = optim.Adam(neg_enc.parameters(), lr=config.lr_2)  # 负编码器的优化器
        neg_dec = LabelSequenceDecoder(data.lab_num + 2, data.fea_num).to(config.device)  # 负解码器
        neg_dec_optim = optim.Adam(neg_dec.parameters(), lr=config.lr_2)  # 负解码器的优化器

        model = Feature2Label(infer_lab.shape[0], data.lab_num, alpha, beta)  # 特征到标签的计算模型

        model.train_val(pos_enc, pos_enc_optim, pos_dec, pos_dec_optim,
                        neg_enc, neg_enc_optim, neg_dec, neg_dec_optim, train_data, val_data)  # 模型训练&验证

        pos_result, neg_result, merge_result = model.infer(data, infer_data, fea)  # 模型推理预测

        pos_results.append(list(pos_result))
        neg_results.append(list(neg_result))
        merge_results.append(list(merge_result))

    mean_std(pos_results, '正编码-解码器')
    mean_std(neg_results, '负编码-解码器')
    mean_std(merge_results, '融合标签方法')

if __name__ == '__main__':
    main('emotions', 10)
