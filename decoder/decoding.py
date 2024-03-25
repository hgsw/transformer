import torch.nn as nn
from utils.util import clones, SublayerConnection, LayerNorm


class DecoderLayer(nn.Module):
    """ 解码层，构造三个子层连接结构
    分别是带有掩码的多头注意力机制、多头注意力机制和前馈全连接"""

    def __init__(self, d_k, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_k = d_k
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(d_k, dropout), 3)

    def forward(self, x, memory, source_mask, target_mask):
        # N层编码器的输入
        m = memory
        # 多头自注意力
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, target_mask))
        # 一般多头自注意力
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, source_mask))
        # 前馈全连接
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    """ 解码器实现，N个解码层DecoderLayer的堆叠"""

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        # self.norm = nn.LayerNorm(layer.d_k)
        self.norm = LayerNorm(layer.d_k)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)

        return self.norm(x)

# d_k = 512
# head = 8
# d_ff = 64
# dropout = 0.2
# c = copy.deepcopy
# attn = MultiHeadAttention(d_k, head)
# ff = PositionalWiseFeedForward(d_k, d_ff, dropout)
# layer = DecoderLayer(d_k, c(attn), c(attn), c(ff), dropout)
# N = 8
# de = Decoder(layer, N)
# x = pe_result
# memory = en_result
# mask = Variable(torch.zeros(8, 4, 4))
# source_mask = target_mask = mask
# de_result = de(x, memory, source_mask, target_mask)
#
# print(de_result)
# print(de_result.shape)
