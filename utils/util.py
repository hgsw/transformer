import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
from torch.autograd import Variable
from pyitcast.transformer_utils import Batch
import numpy as np


def attention(q, k, v, dropout=None, mask=None):
    # 词嵌入维度
    d_k = q.shape[-1]

    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e6)
    score = F.softmax(score, dim=-1)
    if dropout is not None:
        score = dropout(score)
    return torch.matmul(score, v), score


def clones(module, N):
    """
    :param module: 需要复制的网络模块
    :param N: copy数量
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def data_generator(V, batch, num_batch):
    """
    :param V: 随机生成数字
    :param batct: 批次大小
    :param num_batch: 生成样本数
    """
    for i in range(num_batch):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10), dtype=np.int64))
        # 设置起始标志列，解码器进行第一次解码的时候，会使用起始标志列作为输入
        data[:, 0] = 1
        # copy任务中source和target都是一样的，且不需要求梯度
        source = Variable(data, requires_grad=False)
        target = Variable(data, requires_grad=False)
        yield Batch(source, target)


# class LayerNorm(nn.Module):
#     """自定义规范化层"""
#
#     def __init__(self, feature, eps=-1e6):
#         super(LayerNorm, self).__init__()
#         # Parameter 表示参数是模型的参数，需要更新
#         self.a2 = nn.Parameter(torch.ones(feature))
#         self.b2 = nn.Parameter(torch.zeros(feature))
#         self.eps = eps
#
#     def forward(self, x):
#         mean = x.mean(-1, keepdim=True)
#         std = x.std(-1, keepdim=True)
#         return self.a2 * (x - mean) / (std + self.eps) + self.b2


class LayerNorm(nn.Module):
    """torch中规范化层"""

    def __init__(self, feature):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(feature)

    def forward(self, x):
        return self.norm(x)


class SublayerConnection(nn.Module):
    """ 子层连接结构，根据传入的sublayer(示例对象)处理
        在编码层sublayer可以是多头注意机制或者前馈全连接
        在解码层sublayer也可以是带有掩码的多头注意力机制
        SublayerConnection处理流程：规范化 -> 掩码多头/多头/前馈 -> 残差连接"""

    def __init__(self, d_k, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(d_k)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        out = sublayer(self.norm(x))
        out = self.dropout(out)
        return x + out
