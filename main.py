from torch.autograd import Variable
import torch
from pyitcast.transformer_utils import get_std_opt
from pyitcast.transformer_utils import LabelSmoothing
from pyitcast.transformer_utils import SimpleLossCompute
from pyitcast.transformer_utils import run_epoch
from pyitcast.transformer_utils import greedy_decode
from model.transformer import make_model
from utils.util import data_generator

source_vocab = 11
target_vocab = 11
N = 2
V = 11

model = make_model(V, V, N=N)
model_optimizer = get_std_opt(model)
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
loss = SimpleLossCompute(model.generator, criterion, model_optimizer)


def run(model, loss, epochs=50):
    for epoch in range(epochs):
        model.train()
        run_epoch(data_generator(V, 8, 20), model, loss)
        model.eval()
        run_epoch(data_generator(V, 8, 5), model, loss)

    model.eval()
    source = Variable(torch.LongTensor([[1, 10, 2, 5, 4, 6, 7, 8, 9, 10]]))
    # 输入掩码张量，全1表示遮掩
    source_mask = Variable(torch.ones(1, 1, 10))
    result = greedy_decode(model, source, source_mask, max_len=10, start_symbol=1)
    print(result)


if __name__ == "__main__":
    run(model, loss)
