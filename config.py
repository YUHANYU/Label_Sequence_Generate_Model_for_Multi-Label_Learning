# -*-:coding:utf-8-*-

r"""
LSGM模型的参数类
"""

import torch

class Config:
    """

    """
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cpu_or_gpu = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.epochs = 100  # 训练轮次

        self.t_batch_size = 1  # 训练批次数
        self.v_batch_size = 1  # 验证批次数据
        self.i_batch_size = 1  # 测试批次数据

        self.lr_1 = 0.00001  # 模型学习率
        self.lr_2 = 0.000005

        self.sos = 0  # 序列开始符
        self.eos = 1  # 序列结束符
        self.pad = 2  # 序列填充符

        self.fea_norm = 1  # 特征集是否规范化 1-[0, 1]. 2[-1, 1], 0-NO

        self.tree_metric = 'euclidean'  # 查询树的特征衡量距离

        self.enc_layer = 2  # 特征链编码器LSTM层数
        self.enc_bi = True  # 特征链编码器LSTM双向
        self.enc_dropout = 0.1  # 特征链编码器dropout大小

        self.dec_layer = 2  # 标签序列解码器LSTM层数
        self.dec_dropout = 0.1  # 标签序列解码器dropout大小

        self.log = True  # 训练&验证过程写入日志

        self.teach_ratio = 0.5  # 使用教师强制训练机制的比例

        self.gli = True  # 是否加入全局标签信息，使用后，叠加之前预测过的所有标签向量