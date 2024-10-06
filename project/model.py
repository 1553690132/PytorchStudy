import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    net = Net()
    input_v = torch.ones((64, 3, 32, 32))
    output_v = net(input_v)
    print(output_v.shape)
