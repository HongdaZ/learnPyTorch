import random

import torch
from mpmath import arange

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
from matplotlib_inline import backend_inline
def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
import time
# Incrementally plot multiple lines
class Animator:  #@save
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
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
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.2)
animator = Animator()
animator.add(1, 2)
animator.add(3, 4)
for i in range(10):
    animator.add(i, i ** 2)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel = "epoch", xlim = [1, num_epochs],
                        ylim = [.3, .9], legend = ["train loss", "train acc", "test acc"])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < .5, train_loss
    assert train_acc <= 1 and train_acc > .7, train_acc
    assert test_acc <= 1 and test_acc > .7, test_acc

lr = 0.1
def updater(batch_size):
    return sgd([W, b], lr, batch_size)

num_epochs = 10
#train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

def predict_ch3(net, test_iter, n = 6):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    with torch.no_grad():
        preds = get_fashion_mnist_labels(net(X).argmax(axis = 1))
    titles = [true + "\n" + pred for true, pred in zip(trues, preds)]
    show_images(X[0 : n].reshape((n, 28, 28)), 1, n, titles = titles[0 : n])

predict_ch3(net, test_iter)

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = .01)
net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction = "none")
trainer = torch.optim.SGD(net.parameters(), lr = .1)

num_epochs = 20
#train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

x = torch.arange(-8.0, 8.0, 0.1, requires_grad = True)
y = torch.relu(x)

plot(x.detach(), y.detach(), "x", "relu(x)", figsize = (5, 2.5))

y.backward(torch.ones_like(x), retain_graph = True)
plot(x.detach(), x.grad, "x", "grad of relu", figsize = (5, 2.5))

x.grad.zero_()
y = torch.relu(x)
y.sum().backward()
plot(x.detach(), x.grad, "x", "grad of relu", figsize = (5, 2.5))

y = torch.sigmoid(x)
plot(x.detach(), y.detach(), "x", "sigmoid(x)",
     figsize = (5, 2.5))

x.grad.zero_()
y.backward(torch.ones_like(x), retain_graph = True)
plot(x.detach(), x.grad, "x", "grad of sigmoid(x)",
     figsize = (5, 2.5))

y = torch.tanh(x)
plot(x.detach(), y.detach(),"x", "tanh(x)", figsize = (5, 2.5))

x.grad.zero_()
y.backward(torch.ones_like(x))
plot(x.detach(), x.grad, "x", "grad of tanh(x)",
     figsize = (5, 2.5))

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs,
                              num_hiddens, requires_grad = True) * .01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad = True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad = True) * .01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad = True))
params = [W1, b1, W2, b2]

def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X @ W1 + b1)
    return H @ W2 + b2
loss = nn.CrossEntropyLoss(reduction = "none")

num_epochs, lr = 10, .1
updater = torch.optim.SGD(params, lr = lr)
#train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

predict_ch3(net, test_iter)

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = .01)
net.apply(init_weights)

batch_size, lr, num_epochs = 256, .1, 10
loss = nn.CrossEntropyLoss(reduction = "none")
trainer = torch.optim.SGD(net.parameters(), lr = lr)

train_iter, test_iter = load_data_fashion_mnist(batch_size)
#train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0 : 4] = np.array([5, 1.2, -3.4, 5.6])
features = np.random.normal(size = (n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features,
                         np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)
labels = poly_features @ true_w
labels += np.random.normal(scale = .1, size = labels.shape)
ture_w, features, poly_features, labels = [
    torch.tensor(x, dtype = torch.float32) for x in [true_w, features,
                                                     poly_features, labels]
]

def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels,
          num_epochs = 400):
    loss = nn.MSELoss(reduction = "none")
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias = False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = load_array((train_features, train_labels.reshape(-1, 1)),
                            batch_size)
    test_iter = load_array((test_features, test_labels.reshape(-1, 1)),
                           batch_size, is_train = False)
    trainer = torch.optim.SGD(net.parameters(), lr = .01)
    animator = Animator(xlabel = "epoch", ylabel = "loss", yscale = "log",
                        xlim = [1, num_epochs], ylim = [1e-3, 1e2],
                        legend = ["train", "test"])
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    plt.show(block = True)
    print("weight:", net[0].weight.data.numpy())

"""train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])

train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])

train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs = 1500)"""

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * .01, .05
train_data = synthetic_data(true_w, true_b, n_train)
train_iter = load_array(train_data, batch_size)
test_data = synthetic_data(true_w, true_b, n_test)
test_iter = load_array(test_data, batch_size, is_train = False)

def init_params():
    w = torch.normal(0, 1, size = (num_inputs, 1), requires_grad = True)
    b = torch.zeros(1, requires_grad = True)
    return [w, b]

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, .003
    animator = Animator(xlabel = "epochs", ylabel = "loss", yscale = "log",
                        xlim = [5, num_epochs], legend = ["train", "test"])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                         evaluate_loss(net, test_iter, loss)))
    print("w的L2范数是：", torch.norm(w).item())

# train(lambd = 0)
# train(lambd = 10)

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction = "none")
    num_epochs, lr = 100, .003
    trainer = torch.optim.SGD([
        {"params" : net[0].weight, "weight_decay" : wd},
        {"params" : net[0].bias}], lr = lr)
    animator = Animator(xlabel = "epochs", ylabel = "loss", yscale = "log",
                        xlim = [5, num_epochs], legend = ["train", "test"])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (evaluate_loss(net, train_iter, loss),
                          evaluate_loss(net, test_iter, loss)))
    print("w的L2范数：", net[0].weight.norm().item())

# train_concise(3)

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

X = torch.arange(16, dtype = torch.float32).reshape((2, 8))
print(X)
print(dropout_layer(X, 0))
print(dropout_layer(X, .5))
print(dropout_layer(X, 1.))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = .2, .5
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training = True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training == True:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out
net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)

num_epochs, lr, batch_size = 10, .5, 256
loss = nn.CrossEntropyLoss(reduction = "none")
train_iter, test_iter = load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr = lr)

# train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout1),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Dropout(dropout2),
                    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = .01)
net.apply(init_weights)

# trainer = torch.optim.SGD(net.parameters(), lr = lr)
# train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

x = torch.arange(-8.0, 8.0, .1, requires_grad = True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
     legend = ["sigmoid", "gradient"], figsize = (4.5, 2.5))

import hashlib
import os
import tarfile
import zipfile

import requests
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir = os.path.join("..", "data")):
    assert name in DATA_HUB, f"{name} doesn't exist in {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok = True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f"Downloading {fname} from {url}")
    r = requests.get(url, stream = True, verify = True)
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname

def download_extract(name, folder = None):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "只有zip/tar文件可以被解压缩"
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    for name in DATA_HUB:
        download(name)

import numpy as py
import pandas as pd
import torch
from torch import nn

DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download("kaggle_house_test"))

train_data.shape
test_data.shape

all_features = pd.concat((train_data.iloc[:, 1 : -1], test_data.iloc[:, 1 :]))
numeric_features = all_features.dtypes[all_features.dtypes != "object"].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x : (x - x.mean()) / (x.std()))

all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na = True)
all_features.shape

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[: n_train].values.astype("float32"),
                              dtype = torch.float32)
test_features = torch.tensor(all_features[n_train :].values.astype("float32"),
                             dtype = torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1),
                            dtype = torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net

def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float("inf"))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train =None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + i) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                 xlabel="epoch", ylabel="rmse", xlim=[1, num_epochs],
                 legend=["train", "valid"], yscale="log")
            print(f"折{i + 1}，训练log rmse{float(train_ls[-1]):f},",
                  f"验证log rmse{float(valid_ls[-1]):f}")
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f"{k}-折验证：平均训练log rmse:{float(train_l):f},"
      f"平均验证log rmse:{float(valid_l):f}")