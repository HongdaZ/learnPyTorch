import torch
from torch import nn

def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i : i + h, j : j + w] * K).sum()
    return Y


X = torch.tensor([[.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.1]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)