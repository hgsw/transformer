import torch.nn as nn
import torch.nn.functional as F
from utils.util import attention, clones, SublayerConnection, LayerNorm


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_k, head_num, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_k
        self.head_num = head_num
        assert d_k % head_num == 0
        self.head_dim = d_k // head_num

        self.dropout = nn.Dropout(p=dropout)
        self.linears = clones(nn.Linear(d_k, d_k), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(0)
        batch_size = query.size(0)
        # 三个线性层对输入进行进行隐空间特征提取
        query, key, value = \
            [model(x).view(batch_size, -1, self.head_num, self.head_dim).transpose(1, 2) for model, x in
             zip(self.linears, (query, key, value))]
        score, self.attn = attention(query, key, value, dropout=self.dropout, mask=mask)
        score = score.transpose(1, 2).contiguous().view(batch_size, -1, self.head_dim * self.head_num)
        return self.linears[-1](score)

    # 多头注意力机制的另一种实现
    # def forward2(self, query, key, value, mask=None):
    #     if mask is not None:
    #         mask = mask.unsqueeze(0)
    #     batch_size = query.size(0)
    #     query, key, value = \
    #         [model(x).view(batch_size * self.head_num, -1, self.head_dim) for model, x in
    #          zip(self.linears, (query, key, value))]
    #     score, self.attn = attention(query, key, value, dropout=self.dropout, mask=mask)
    #     score = score.view(batch_size, -1, self.head_dim * self.head_num)
    #     return self.linears[-1](score)


class PositionalWiseFeedForward(nn.Module):
    """前馈全连接"""

    def __init__(self, d_k, hidden_size, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()

        self.w1 = nn.Linear(d_k, hidden_size)
        self.w2 = nn.Linear(hidden_size, d_k)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.w1(x)
        out = F.relu(out)
        out = self.dropout(out)
        return self.w2(out)


class EncoderLayer(nn.Module):
    """ 编码层，构造两个子层连接结构，分别是多头注意力机制和前馈全连接"""

    def __init__(self, d_k, attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        # 拷贝2个子层连接结构，具体处理方式(多头/前馈)调用时指定
        self.sublayer = clones(SublayerConnection(d_k, dropout), 2)
        # 保存词嵌入维度，方便后续使用
        self.size = d_k

    def forward(self, x, mask):
        """ 先走多头注意力机制，在过前馈全连接。 Transformer编码顺序"""
        x = self.sublayer[0](x, lambda x: self.attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    """ 编码器实现，N个编码层EncoderLayer的堆叠"""

    def __init__(self, encoder_layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(encoder_layer, N)
        # 使用自定义规范会层 encoder_layer.size 词嵌入维度
        self.norn = LayerNorm(encoder_layer.size)
        # torch中规范会层
        # self.norn = nn.LayerNorm(encoder_layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norn(x)

# d_k = 512
# head = 8
# d_ff = 64
# x = pe_result
# dropout = 0.2
# N = 8
# c = copy.deepcopy
# mask = Variable(torch.zeros(8, 4, 4))
#
# self_attn = MultiHeadAttention(d_k, head)
# ff = PositionalWiseFeedForward(d_k, d_ff, dropout)
# layer = EncoderLayer(d_k, c(self_attn), c(ff), dropout)
#
# en = Encoder(layer, N)
# en_result = en(x, mask)
# print(en_result)
# print(en_result.shape)
