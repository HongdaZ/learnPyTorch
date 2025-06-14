import torch
from torch import nn


import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from torch.nn import functional as F
from d2l import train_ch6
from d2l import try_gpu

net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size = 11, stride = 4, padding = 1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2),
    nn.Conv2d(96, 256, kernel_size = 5, padding = 2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2),
    nn.Conv2d(256, 384, kernel_size = 3, padding = 1),
    nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size = 3, padding = 1),
    nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size = 3, padding = 1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2),
    nn.Flatten(),
    nn.Linear(6400, 4096),
    nn.ReLU(),
    nn.Dropout(p = .5),
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Dropout(p = .5),
    nn.Linear(4096, 10)
)
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, "out shape:\t", X.shape)

batch_size = 128
from d2l import load_data_fashion_mnist
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize = 224)

lr, num_epochs = .01, 10
# train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size = 3, padding = 1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*conv_blks, nn.Flatten(),
                         nn.Linear(out_channels * 7 * 7, 4096),
                         nn.ReLU(), nn.Dropout(.5),
                         nn.Linear(4096, 4096),
                         nn.ReLU(), nn.Dropout(.5),
                         nn.Linear(4096, 10))
net = vgg(conv_arch)
X = torch.randn(size = (1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, "output shape:\t", X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)
lr, num_epochs, batch_size = .05, 10,  128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize = 224)
# train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 1),
        nn.ReLU()
    )

net = nn.Sequential(
    nin_block(1, 96, kernel_size = 11, strides = 4, padding = 0),
    nn.MaxPool2d(3, stride = 2),
    nin_block(96, 256, kernel_size = 5, strides = 1, padding = 2),
    nn.MaxPool2d(3, stride = 2),
    nin_block(256, 384, 3, 1, 1),
    nn.MaxPool2d(3, stride = 2),
    nn.Dropout(.5),
    nin_block(384, 10, kernel_size = 3, strides = 1, padding = 1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)

X = torch.rand(size = (1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, "output shape:\t", X.shape)

lr, num_epochs, batch_size = .1, 10, 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize = 224)
# train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size = 1)
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size = 1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size = 3, padding = 1)
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size = 1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size = 5, padding = 2)
        self.p4_1 = nn.MaxPool2d(kernel_size = 3, stride =1, padding = 1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size = 1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        return torch.cat((p1, p2, p3, p4), dim =1)

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1,1)),
                   nn.Flatten())

net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

X = torch.rand(size=(1, 1, 96, 96))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)

lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
# train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())


class A:
    def age(self):
        print("Age is 21")


class B:
    def age(self):
        print("Age is 23")


class C(A, B):
    def age(self):
        super(C, self).age()


class D(A, B):
    def age(self):
        super(A, self).age()


c = C()
d = D()
c.age()
d.age()

print(C.__mro__)
print(D.mro())

from d2l import BatchNorm
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
# train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

net[1].gamma.reshape((-1, )), net[1].beta.reshape(-1,)

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size = 5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(6, 16, kernel_size = 5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size = 2, stride = 2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

# train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

from d2l import Residual

blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape

blk = Residual(3, 6, use_1x1conv = True,
               strides = 2)
blk(X).shape

b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size = 7, stride = 2,
                             padding =3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
from d2l import resnet_block
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block = True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(), nn.Linear(512, 10))
X = torch.rand(size = (1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, "output shape:\t", X.shape)

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
# train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

from d2l import conv_block, DenseBlock, transition_block
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
Y.shape

blk = transition_block(23, 10)
blk(Y).shape

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

num_channels, growth_rate = 64, 32
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加一个转换层，使通道数量减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2
net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))

lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())