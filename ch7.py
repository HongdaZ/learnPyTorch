import torch
from torch import nn


import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
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
train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())