import torch
import torch.nn as nn
import math
import numpy as np
from torch.autograd import Variable


class Embeddings(nn.Module):
    """构建embedding类实现文本嵌入"""

    def __init__(self, d_model, vocab):
        # d_model: 词嵌入维度
        # vocab: 词表的大小
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# class PositionalEncoding(nn.Module):
#     """位置编码"""
#
#     def __init__(self, d_model, pad_size=5000):
#         # d_model 词嵌入维度
#         # pad_size 默认词汇大小
#         super(PositionalEncoding, self).__init__()
#         self.d_model = d_model
#         self.pad_size = pad_size
#         pe = torch.zeros(pad_size, d_model)
#
#         for t in range(pad_size):
#             for i in range(d_model // 2):
#                 angle_rate = 1 / (10000 ** (2 * i // d_model))
#                 pe[t, 2 * i] = np.sin(t * angle_rate)
#                 pe[t, 2 * i + 1] = np.cos(t * angle_rate)
#
#         # # 双层循环等价写法
#         # pe = torch.tensor(
#         #     [[pad / (10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)] for pad in range(pad_size)])
#         #
#         # pe[:, 0::2] = np.sin(pe[:, 0::2])
#         # pe[:, 1::2] = np.cos(pe[:, 1::2])
#         # 将位置编码扩展到三维
#         pe = pe.unsqueeze(0)
#         # 将位置编码矩阵注册成模型的buffer，buffer不是模型的参数，不跟随优化器更新
#         # 注册成buffer后，在模型保存后重新加载模型的时候，将这个位置编码将和参数一起加载进来
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         # 位置编码不需要反向更新
#         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
#         return x


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        # d_model: 词嵌入的维度
        # dropout: 置零率
        # max_len: 每个句子的最大长度
        super(PositionalEncoding, self).__init__()
        # 实例化Dropout层
        # self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码矩阵，大小max_len * d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵 大小max_len * 1
        position = torch.arange(0, max_len).unsqueeze(1)
        # 定义变换矩阵, 以2等差到d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

        # 将变换矩阵进行奇数偶数分别赋值
        # pe的所有偶数列
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe的维度是max_len * d_model将pe扩充为三维张量
        pe = pe.unsqueeze(0)

        # 将位置编码矩阵注册成模型的buffer，buffer不是模型的参数，不跟随优化器更新
        # 注册成buffer后，在模型保存后重新加载模型的时候，将这个位置编码器和参数加载进来
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: 是三维的文本序列的词嵌入表示
        # pe 编码矩阵太长，将第二个维度，也是max_len的维度缩小成x句子长度
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return x


d_model = 512
vocab = 1000
x = Variable(torch.LongTensor([[12, 344, 123, 44], [432, 677, 97, 12]]))
embedding = Embeddings(d_model, vocab)
embr = embedding(x)

x_pe = embr
pe = PositionalEncoding(d_model)
pe_result = pe(x_pe)
# print(pe_result)
