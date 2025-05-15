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