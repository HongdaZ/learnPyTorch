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

