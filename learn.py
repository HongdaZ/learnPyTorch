import sys
print(sys.version)
print("hello, world!")
import random
if 5 > 2:
    print("Five is greater than two!")
x = 5
y = "hello, world!"
# This is a comment
"""
This is a comment writen in multiple lines
"""
x = str(3)
y = int(3)
z = float(3)
print(type(x))

fruits = ("apple", "banana", "cherry")
x, y, z = fruits

x, y, z = "orange", "banana", "cherry"
x = y = z = "orange"
y = str(1)
x = "Python is awesome"
print(x)
print(x + y + z)
print(x, y, z)
x = "awesome"
def myfunc():
    global x
    x = "hello"
    print("Python is " + x)
myfunc()
type(x)
x = bool(5)
x = bytes(5)
x = float(1)
x + 1e-32
y = 1e-32
x += y
y = 3.6
int(y)
import random
print(random.randrange(1, 10))
for x in z:
    print(x)
a = "hello, world!"
print(a[1])
for i in range(0, 5):
    print(a[i])
print(a[2 : 5])
print(a[-5 : -2])
price = 25
txt = f"The price is {price : .2f} dollars"
print(txt)
x = 200
print(isinstance(x, int))
y = x
x is y
x
y = 3
x is y
x = [1, 3, 4]
y = x
y is x
y[2] = 1
y is x
y = x.copy()

thislist = ["apple", "banana", "cherry", "orange",
            "kiwi", "melon", "mango"]
print(thislist[-4 : -1])

a = ("a", "b", "c", "d", "e", "f", "g", "h")
x = slice(3, 5)
print(a[x])
thislist = ["apple", "banana", "cherry"]
thislist[1:2] = ["blackcurrant", "watermelon"]
print(thislist)

thislist = ["apple", "banana", "cherry"]
thislist[1:3] = ["watermelon"]
print(thislist)
thislist = ["apple", "banana", "cherry"]
thislist.insert(2, "watermelon")
print(thislist)
tropical = ["mango", "pineapple", "papaya"]
thislist.extend(tropical)
thislist.remove("banana")
thislist.pop(2)
del thislist[0]
del thislist
thislist = ["apple", "banana", "cherry"]
thislist.clear()
thislist = ["apple", "banana", "cherry"]
newlist = [x.upper() for x in thislist if "a" in x]
newlist.sort(reverse = True)
def myFunc(n):
    return abs(n - 50)
thislist = [100, 50, 65, 82, 23]
thislist.sort(key = myFunc)
mylist = thislist.copy()
mylist = list(("hello", "world"))
newlist = mylist[:]
clist = mylist + newlist
clist.extend(clist)

import torch, numpy
from dask.dataframe.dispatch import tolist
from docutils.nodes import legend, inline

x = torch.arange(12)
x
x.shape
x = x.reshape(3, 4)
torch.randn(3, 4)
x = torch.tensor([1, 2, 4, 8])
y = torch.tensor([2, 2, 2, 4])
x + y
x = torch.arange(12, dtype = torch.float32).reshape(3, 4)
y = torch.randn(12, dtype = torch.float32).reshape(3, 4)
z = torch.cat((x, y), dim = 0)
o = torch.cat((x, y), dim = 1)

x == y
x.sum()
x[1 : 3]
id(y)
y = x + y
id(y)
z = torch.zeros_like(y)
id(z)
z[:] = x + y
id(z)
id(x)
x += y
id(x)
x = y
id(x)
id(y)
z = x
id(z)
a = x.numpy()
type(a)
import os
os.makedirs('data', exist_ok = True)
data_file = os.path.join('data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('2,NA,106000\n')
    f.write('NA,Pave,127500\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
import pandas as pd
data = pd.read_csv(data_file)
print(data)
inputs, mid, outputs = data.iloc[:, 0:1], data.iloc[:, 1], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
mid = pd.get_dummies(mid, dummy_na = True, dtype = 'int')
inputs = pd.DataFrame.join(inputs, mid)
x = torch.tensor(inputs.values, dtype = torch.float32)
y = torch.tensor(outputs.values, dtype = torch.float32)
x = torch.tensor(3.0)
y = torch.tensor(2.0)
x + y, x ** y, x * y, x / y
x = torch.arange(3)
len(x)
x.shape
A = torch.arange(20).reshape(5, 4)
A[2, 3]
A.T
x = torch.arange(24).reshape(2, 3, 4)
A = torch.arange(20, dtype = torch.float32).reshape(5, 4)
B = A.clone()
id(A)
id(B)
A * B
C = A
A * C
A.sum(axis = 0)
torch.sum(A[:, 0])
A.mean()
A.sum() / A.numel()
A.mean(axis = 0), A.sum(axis = 0) / A.shape[0]
A.cumsum(axis = 0)
y = torch.ones(4, dtype = torch.float32)
x = torch.arange(4, dtype = torch.float32)
torch.dot(x, y)
A.shape
torch.mv(A, y)
B = torch.randn(4, 3)
torch.mm(A, B)
torch.norm(x)
torch.abs(x).sum()
import numpy as np
import matplotlib.pyplot  as plt
from matplotlib_inline import backend_inline
def f(x):
    return 3 * x ** 2 - 4 * x
f(11)

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h
h = 0.1
for i in range(5):
    print(f'h = {h : .5f}, numerical limit = {numerical_lim(f, 1, h) : .5f}')
    h *= .1

def use_svg_display():
    backend_inline.set_matplotlib_formats('svg')
def set_figsize(figsize = (3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def has_one_axis(X):
    return (hasattr(X, 'ndim') and X.ndim == 1 or isinstance(X, list)
            and not hasattr(X[0], '__len__'))

def plot(X, Y = None, xlabel = None, ylabel = None, legend = None,
         xlim = None, ylim = None, xscale = 'linear', yscale = 'linear',
         fmts = ('-', 'm--', 'g-', 'r:'), figsize = (35, 25), axes = None):
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        axes.cla()
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)
        set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        plt.show(block = True)
x = np.arange(0, 3, 0.1)
plt.rcParams.update({'font.size': 22}) # change font size
plot(x, [f(x), 2 * x -3], 'x', 'f(x)', legend = ['f(x)', 'Tangent line(x = 1)'])

xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])

plt.plot(xpoints, ypoints)
plt.show(block = True)
x = torch.arange(4.0, dtype = torch.float32)
x.requires_grad_(True)
x.grad
y = 2 * torch.dot(x, x)
y.backward()
x.grad
x.grad == 4 * x
y = x.sum()
x.grad.zero_()
y.backward()
x.grad
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
x.grad == u
x.grad.zero_()
v = y.clone()
z = v * x
z.sum().backward()
x.grad == v

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size = (), requires_grad = True)
d = f(a)
d.backward()
a.grad == d / a
a.grad.zero_()

torch.ones(4, dtype = torch.float32)


import math
import time
import numpy as np
import torch
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
        return np.array(self.times).cumsum(),tolist()

c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'

timer.start()
d = a + b
f'{timer.stop():.5f} sec'

def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)
x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]
y = [normal(x, mu, sigma) for mu, sigma in params]
plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel = "x",
     ylabel = 'PDF', figsize = ( 45, 25), legend = [f'mean {mu}, std {sigma}'
                                                     for mu , sigma in params])
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y +=torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)
plt.show(block = True)
import random
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
w = torch.normal(0, 0.01, size = (2, 1), requires_grad= True)
b = torch.zeros(1, requires_grad = True)

def linreg(X, w, b):
    return torch.matmul(X, w) + b
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
w = torch.normal(0, 0.01, size = (2, 1), requires_grad= True)
b = torch.zeros(1, requires_grad = True)
lr = .01
num_epochs = 10
net = linreg
loss = squared_loss
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f"epoch {epoch + 1}, loss {float(train_l.mean()): f}")
true_w - w.detach().reshape(true_w.shape)
true_b - b.detach()
import numpy as np
import torch
from torch.utils import data

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, b, 1000)
def load_array(data_arrays, batch_size, is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle = is_train)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

for X, y in data_iter:
    print(X)
    print(y)
    break

next(iter(data_iter))

from torch import nn
net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr = 0.03)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    with torch.no_grad():
        l = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {l : f}")

   w = net[0].weight.data
   true_w - w.reshape(true_w.shape)
   b = net[0].bias.data
   true_b - b

import torch
import torchvision
from torch.utils import data
from torchvision import transforms
use_svg_display()
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root = ".data", train = True, transform = trans, download = True)
mnist_test = torchvision.datasets.FashionMNIST(
    root = "data", train = False, transform = trans, download = True
)

len(mnist_train), len(mnist_test)
mnist_train[0][0].shape

def get_fashion_mnist_labels(labels):
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal",
                   "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]
labels = range(1, 10)
get_fashion_mnist_labels(labels)

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
        if(titles):
            ax.set_title(titles[i])
    plt.show(block = True)
    return(axes)
X, y = next(iter(data.DataLoader(mnist_train, batch_size = 18)))
show_images(X.reshape(18, 28, 28), 2, 9,
            titles = get_fashion_mnist_labels(y))
a = ("John", "Charles", "Mike")
b = ("Jenny", "Christy", "Monica")

x = zip(a, b)
class MyNumbers:
    def __iter__(self):
        self.a = 1
        return self
    def __next__(self):
        if self.a <= 20:
            x = self.a
            self.a += 1
            return x
        else:
            raise StopIteration
myclass = MyNumbers()
myiter = iter(myclass)
for x in myiter:
    print(x)