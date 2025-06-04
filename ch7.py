import torch
from torch import nn


import torchvision
from torch.utils import data
from torchvision import transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline

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