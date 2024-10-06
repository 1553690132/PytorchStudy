import torch
import torch.nn.functional as F

x = torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                  [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

k = torch.tensor([[[0, 1], [2, 3]],
                  [[1, 2], [3, 4]]])

x = torch.reshape(x, (1, 2, 3, 3))
k = torch.reshape(k, (1, 2, 2, 2))

res = F.conv2d(x, k, padding=0, stride=1)
print(res)

x2 = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
x2 = torch.reshape(x2, (1, 1, 3, 3))
print(x2)
y2 = torch.nn.MaxPool2d((2, 2), stride=1)(x2)
print(y2)
