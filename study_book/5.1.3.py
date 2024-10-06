import torch
from torch.nn import Conv2d


def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    print(Y)
    return Y


X = torch.ones(6, 8)
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1, -1]])
print(K)

corr2d(X, K)
X = torch.reshape(X, (1, 1, 6, 8)).float()
print(X)
K = torch.reshape(K, (1, 1, 1, 2)).float()
print(K)

print(torch.nn.functional.conv2d(X, K))

