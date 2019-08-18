# -*-coding;utf-8-*-

r"""
LSGM模型的主函数
"""

from torch.utils.data import DataLoader
from torch import optim

from data.process import Process
from lsgm.encoder import FeatureChainEncoder
from lsgm.decoder import LabelSequenceDecoder
from lsgm.feature2label import Feature2Label

from config import Config
config = Config()


def main(data_type, k):
    print('多标记数据类型{}，最近邻个数{}。'.format(data_type, k))
    data = Process(data_type, k)  # 数据对象

    fea, lab = data.get_data()  # 获取特征集，标签集

    lab_new = data.sort_lab_order(lab, 'd')  # 降序排列标签集

    train_fea, train_lab, val_fea, val_lab, infer_fea, infer_lab = data.t_v_i(fea, lab_new.copy())  # 训练，验证和推理的x，y数据

    train_k, val_k, infer_k = data.get_k_ins(train_fea, val_fea, infer_fea)  # 查询训练，验证和测试中每示例的k个最近邻对象

    alpha, beta = data.get_alpha_beta(infer_k, lab_new, k)  # 获得推理集的标签先验概率alpha和beta

    train_chain, val_chain, infer_chain = data.build_fea_chain('mix', train_k, val_k, infer_k)  # 构建训练，验证和推理特征链

    # 构建训练特征链-正负标签序列，验证特征链-正负标签序列，推理特征链-标签序列
    train_chain_lab, val_chain_lab, infer_chain_lab = data.merge_chain_lab(train_chain, val_chain, infer_chain, lab_new)

    train_data = DataLoader(train_chain_lab, config.t_batch_size, shuffle=False, drop_last=False)  # 训练数据
    val_data = DataLoader(val_chain_lab, config.v_batch_size, shuffle=False, drop_last=False)  # 验证数据
    infer_data = DataLoader(infer_chain_lab, config.i_batch_size, shuffle=False, drop_last=False)  # 推理数据

    pos_enc = FeatureChainEncoder(data.ins_num, data.fea_num, fea).to(config.device)  # 负编码器
    pos_enc_optim = optim.Adam(pos_enc.parameters(), lr=config.lr_1)  # 负编码器的优化器
    pos_dec = LabelSequenceDecoder(data.lab_num + 2, data.fea_num).to(config.device)  # 负解码器
    pos_dec_optim = optim.Adam(pos_dec.parameters(), lr=config.lr_1)  # 负解码器的优化器

    neg_enc = FeatureChainEncoder(data.ins_num, data.fea_num, fea).to(config.device)  # 负编码器
    neg_enc_optim = optim.Adam(neg_enc.parameters(), lr=config.lr_2)  # 负编码器的优化器
    neg_dec = LabelSequenceDecoder(data.lab_num + 2, data.fea_num).to(config.device) # 负解码器
    neg_dec_optim = optim.Adam(neg_dec.parameters(), lr=config.lr_2)  # 负解码器的优化器

    model = Feature2Label(data.lab_num)  # 特征到标签的计算模型

    # model.train_val(pos_enc, pos_enc_optim, pos_dec, pos_dec_optim,
    #                 neg_enc, neg_enc_optim, neg_dec, neg_dec_optim, train_data, val_data)  # 模型训练&验证

    model.infer(data, infer_data, fea)


if __name__ == '__main__':
    main('yeast', 10)
