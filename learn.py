import random

import torch
torch.cuda.is_available()
# autograd
x = torch.arange(4.0)
x.requires_grad_(True)
y = 2 * torch.dot(x, x)
y.backward()
x.grad
x.grad.zero_()
y = x * x
sum(y).backward()
x.grad

x.grad.zero_()
y = x * x
u = y.detach()
z = x * u
sum(z).backward()
x.grad
x.grad.zero_()

y.sum().backward()
x.grad

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c =  1000 * b
    return c
a = torch.randn(size = (), requires_grad = True)
d= f(a)
d.backward()
a.grad == d / a

from torch.distributions import multinomial

import matplotlib.pyplot as plt
import matplotlib

fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim = 0)
estimates = cum_counts / cum_counts.sum(dim = 1, keepdim = True)
estimates


for i in range(6):
    plt.plot(estimates[:, i].numpy(),
                 label = ("P(die = " + str(i + 1) + ")"))

plt.gca().set_xlabel("Groups of experiments")
plt.gca().set_ylabel("Estimated probability")
plt.legend()
plt.show(block = True)

help(torch.ones)
torch.ones((3, 4), dtype = torch.float32)

import math
import time
import numpy as np

n = 10000
a = torch.ones(n)
b = torch.ones(n)
class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        self.tik = time.time()
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    def avg(self):
        return sum(self.times) / len(self.times)
    def sum(self):
        return sum(self.times)
    def cumsum(self):
        return np.array(self.times).cumsum().tolist()
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'

timer.start()
d = a + b
f'{timer.stop():.5f} sec'

def normal(x, mu, sigma):
    p = 1/ math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)

x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]
for mu, sigma in params:
    plt.plot(x, normal(x, mu, sigma), label =("Std = " + f'{sigma: .2f}') )
plt.legend()
plt.show(block = True)

import torch

def synthetic_data(w, b, num_example):
    x = torch.normal(0, 1, (num_example, len(w)))
    y = torch.matmul(x, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return x, y.reshape((-1, 1))

true_w = torch.tensor((2, -3.4))
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

#plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
#plt.show(block = True)

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i : min(i + batch_size,
                                                     num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for x, y in data_iter(batch_size, features, labels):
    print(x, "\n", y)
    break
data_gen = data_iter(batch_size, features, labels)
iter_data = iter(data_gen)
next(iter_data)
def number_generator(n):
    i = 0
    while i < n:
        yield i
        i += 1
my_generator = number_generator(5)

iter_num = iter(my_generator)
next(iter_num)
print(next(my_generator))
print(next(my_generator))

for num in my_generator:
    print(num)

def linreg(x, w, b):
    return torch.matmul(x, w) + b
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

"""Mutable values can change in plance while immutable values refer to
another object after assignment."""
def modify_list(my_list):
    my_list.append(4)  # Modifies the original list
    my_list = [5, 6, 7] # Reassigns the parameter, does not affect the original

original_list = [1, 2, 3]
modify_list(original_list)
print(original_list) # Output: [1, 2, 3, 4]

lr = 0.03
num_epochs = 4
net = linreg
loss = squared_loss

w = torch.tensor([0, 0],dtype= torch.float32, requires_grad = True)
b = torch.zeros(1, requires_grad = True)
for epoch in range(num_epochs):
    for x, y in data_iter(batch_size, features, labels):
        l = loss(net(x, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch{epoch + 1}, loss = {float(train_l.mean()):f}')


import numpy as np
import torch
from torch.utils import data
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
##
def add(x, y, z):
    return x + y + z

numbers = [1, 2, 3]
result = add(*numbers) # Equivalent to add(1, 2, 3)
print(result) # Output: 6
##
def my_function(*args):
    for arg in args:
        print(arg)

my_function(1, 2, 3, "hello") # Output: 1, 2, 3, hello

def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr = 0.02)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        l = loss(net(X), y)
        print(f'epoch {epoch + 1}, loss {l: f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

import torch
import torchvision
from torch.utils import data
from torchvision import transforms

trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root = "../data", train = True, transform = trans, download = True)
mnist_test = torchvision.datasets.FashionMNIST(
    root = "../data", train = False, transform = trans, download = True)

len(mnist_train)
len(mnist_test)

mnist_train[0][0].shape

def get_fashion_mnist_labels(labels):
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal",
                   "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize = figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

X, y = next(iter(data.DataLoader(mnist_train, batch_size = 18)))
show_images(X.reshape(18, 28, 28), 2, 9,
            titles = get_fashion_mnist_labels(y))
plt.show(block = True)
batch_size = 256
def get_dataloader_worker():
    return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle = True,
                             num_workers = get_dataloader_worker())
timer = Timer()
for X, y in train_iter:
    continue
f"{timer.stop() : .2f} sec"

def load_data_fashion_mnist(batch_size, resize = None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train = True, transform = trans, download = False)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train = False, transform = trans, download = True)
    return (data.DataLoader(mnist_train, batch_size, shuffle = True,
                            num_workers = get_dataloader_worker()),
            data.DataLoader(mnist_test, batch_size, shuffle = True,
                            num_workers = get_dataloader_worker()))

train_iter, test_iter = load_data_fashion_mnist(32, resize = 64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

import torch
from IPython import display
batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10
W = torch.normal(0, 0.01, size = (num_inputs, num_outputs),
                 requires_grad = True)
b = torch.zeros(num_outputs, requires_grad = True)
def  softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim = True)
    return X_exp / partition

X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)

def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

y = torch.tensor([0, 2])
y_hat  = torch.tensor([[.1, .3, .6], [.3, .2, .5]])
y_hat[[0, 1], y]
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] >1:
        y_hat = y_hat.argmax(axis = 1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def  __getitem__(self, idx):
        return self.data[idx]
evaluate_accuracy(net, test_iter)

def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    def __init__(self, xlabel = None, ylabel = None, legend = None,
                 xlim = None, ylim = None, xscale = "linear", yscale = "linear",
                 fmts = ("-", "m--", "g-.", "r:"), nrows = 1, ncols = 1,
                 figsize = (3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize = figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        plt.gca().set_xlabel(xlabel)
        plt.gca().set_xlim(xlim)
        plt.gca().set_ylim(ylim)
        plt.legend(legend)
        self.X, self.Y, self.fmts = None, None, fmts
    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.Y[i].append(b)
                self.X[i].append(a)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        display.display(self.fig)
        display.clear_output(wait = True)
        plt.show(block = True)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel = "epoch", xlim = [1, num_epochs],
                        ylim = [.3, .9], legend = ["train loss", "train acc", "test acc"])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
    train_loss, train_acc = train_metrics
    #assert train_loss < .5, train_loss
    #assert train_acc <= 1 and train_acc > .7, train_acc
    #assert test_acc <= 1 and test_acc > .7, test_acc

lr = .1
def updater(batch_size):
    return sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)