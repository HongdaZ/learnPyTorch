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
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

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
torch.ones((3, 4))

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
    plt.plot(x, normal(x, mu, sigma))
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

plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
plt.show(block = True)

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

def number_generator(n):
    i = 0
    while i < n:
        yield i
        i += 1
my_generator = number_generator(5)
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
lr = 0.03
num_epochs = 30
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
        print(f'epoch{epoch + 1}, loss = { float(train_l.mean()):f}')

