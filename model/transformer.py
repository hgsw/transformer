import torch.nn as nn
import copy
from encoder.encoding import MultiHeadAttention, PositionalWiseFeedForward, Encoder, EncoderLayer
from decoder.decoding import Decoder, DecoderLayer
from embedding.embeddings import Embeddings, PositionalEncoding
from output.generator import Generator


class EncoderDecoder(nn.Module):
    """ Transformer架构搭建，词嵌入 -> 编码器 -> 解码器 -> 概率输出"""

    def __init__(self, encoder, decoder, source_embed, target_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, source, target, source_mask, target_mask):
        encoder_result = self.encode(source, source_mask)
        return self.decode(encoder_result, source_mask, target, target_mask)

    def encode(self, source, source_mask):
        # 编码词嵌入和输出
        return self.encoder(self.src_embed(source), source_mask)

    def decode(self, memory, source_mask, target, target_mask):
        # 解码词嵌入和输出
        # memory 编码器输出的张量
        return self.decoder(self.tgt_embed(target), memory, source_mask, target_mask)


# vocab_size = 1000
# d_k = 512
# encoder = en
# decoder = de
# generator = gen
# source_embed = nn.Embedding(vocab_size, d_k)
# target_embed = nn.Embedding(vocab_size, d_k)
# mask = Variable(torch.zeros(8, 4, 4))
# source_mask = target_mask = mask
# source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
# ed = EncoderDecoder(encoder, decoder, source_embed, target_embed, generator)
# ed_result = ed(source, target, source_mask, target_mask)
# print(ed_result)
# print(ed_result.shape)


def make_model(source_vocab, target_vocab, N=8, d_k=512, d_ff=2048, head=8, dropout=0.0):
    c = copy.deepcopy
    attn = MultiHeadAttention(d_k, head)
    ff = PositionalWiseFeedForward(d_k, d_ff, dropout)
    position = PositionalEncoding(d_k)
    model = EncoderDecoder(Encoder(EncoderLayer(d_k, c(attn), c(ff), dropout), N),
                           Decoder(DecoderLayer(d_k, c(attn), c(attn), c(ff), dropout), N),
                           nn.Sequential(Embeddings(d_k, source_vocab), c(position)),
                           nn.Sequential(Embeddings(d_k, target_vocab), c(position)),
                           Generator(d_k, target_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)
    return model
