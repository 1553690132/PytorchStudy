import torch
import torch.nn as nn


class FancyMLP(nn.Module):
    def __init__(self):
        super(FancyMLP, self).__init__()

        self.rand_weight = torch.rand((20, 20), requires_grad=False)  # 不可训练参数
        self.linear = nn.Linear(20, 20)

    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.mm(x, self.rand_weight.data) + 1)

        while x.norm().item() > 1:
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x


X = torch.rand(2, 20)
net = FancyMLP()
print(net)
net(X)
print(X)
