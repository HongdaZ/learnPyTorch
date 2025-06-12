import torch
from IPython.core.pylabtools import figsize
from torch import nn
import matplotlib.pyplot as plt
import random
T = 1000
time = torch.arange(1, T + 1, dtype = torch.float32)
x = torch.sin(.01 * time) + torch.normal(0, .2, (T,))
plt.show(block = True)
plt.plot(time, x)

tau = 4
features = torch.zeros((T - tau, tau))
for i in range(tau):
    features[:, i] = x[i : T - tau + i]
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600

from d2l import load_array
train_iter = load_array((features[: n_train], labels[: n_train]),
                        batch_size, is_train = True)
from d2l import get_net
from d2l import evaluate_loss
loss = nn.MSELoss(reduction = "none")
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f"epoch {epoch + 1},",
              f"loss: {evaluate_loss(net, train_iter, loss):f}")

net = get_net()
train(net, train_iter, loss, 5, .01)
from d2l import plot
onestep_preds = net(features)
plot([time, time[tau :]],
         [x.detach().numpy(), onestep_preds.detach().numpy()], )
plt.show(block = True)

multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau : i].reshape((1, -1)))

plot([time, time[tau :], time[n_train + tau :]],
     [x.detach().numpy(), onestep_preds.detach().numpy(),
      multistep_preds[n_train + tau :].detach().numpy()],
     "time", "x", legend = ["data", "1-step preds", "multistep preds"],
     xlim = [1, 1000], figsize = (6, 3))
plt.show(block = True)

max_steps = 64
features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
for i in range(tau):
    features[:, i] = x[i : i + T - tau - max_steps + 1]
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau : i]).reshape(-1)
steps = (1, 4, 16, 64)
plot([time[tau + i - 1 : T - max_steps + i] for i in steps],
     [features[:, (tau + i - 1)].detach().numpy() for i in steps], "time", "x",
     legend = [f"{i}-step preds" for i in steps], xlim = [5, 1000],
     figsize = (6, 3))

import collections
import re
from d2l import DATA_HUB, DATA_URL, download

DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """将时间机器数据集加载到文本行的列表中"""
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])
print(lines[10])

def tokenize(lines, token = "word"):
    if token == "word":
        return [line.split() for line in lines]
    elif token == "char":
        return [list(line) for line in lines]
    else:
        print("Error: unknown token type:" + token)
tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

for i in [0, 10]:
    print("文本:", tokens[i])
    print("索引:", vocab[tokens[i]])

def load_corpus_time_machine(max_tokens = -1):
    lines = read_time_machine()
    tokens = tokenize(lines, "char")
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[: max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)

tokens = tokenize(read_time_machine())
corpus = [token for line in tokens for token in line]
vocab = Vocab(corpus)
vocab.token_freqs[: 10]

freqs = [freq for token, freq in vocab.token_freqs]
plot(freqs, xlabel = "token: x", ylabel = "freq: n(x)",
     xscale = "log", yscale = "log")

bigram_tokens = [pair for pair in zip(corpus[: -1], corpus[1 :])]
bigram_vocab = Vocab(bigram_tokens)
bigram_vocab.token_freqs[: 10]

trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]

bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])

def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中，
    # 来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

import random
x = list(range(10))
random.shuffle(x)

def seq_data_iter_sequential(corpus, batch_size,  num_steps):
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset : offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1 : offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield X, Y

for X, Y in seq_data_iter_sequential(my_seq, batch_size = 2, num_steps = 5):
    print("X: ", X, "Y: ", Y)

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps
    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter = False, max_tokens = 10000):
    data_iter = SeqDataLoader(batch_size, num_steps,
                              use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

class my_gen:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __iter__(self):
        i = self.start
        while i < self.end:
            yield i
            i += 1

gen1 = my_gen(2, 4)
for i in gen1:
    print(i)

def myFunc():
  yield "Hello"
  yield 51
  yield "Good Bye"

x = myFunc()

for z in x:
  print(z)

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4 ,4))
res1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
res2 = torch.matmul(torch.cat((X, H), 1),
                    torch.cat((W_xh, W_hh), 0))
torch.isclose(res1, res2)

import math
from torch.nn import functional as F
batch_size, num_steps = 32, 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)
F.one_hot(torch.tensor([0, 2]), len(vocab))
X = torch.arange(10).reshape((2, 5))
F.one_hot(X.T, 28).shape

