import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """ 输出层，经过线性全连接后接softmax转化为概率形式"""

    def __init__(self, d_k, vocab_size):
        super(Generator, self).__init__()
        self.d_k = d_k
        self.vocab_size = vocab_size
        self.project = nn.Linear(d_k, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)
        # return F.softmax(self.project(x), dim=-1)
