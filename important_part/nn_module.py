import torch

from torch import nn


class LWH(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


test = LWH()

x = torch.tensor([1, 2, 3])
output = test(x)
print(output)
