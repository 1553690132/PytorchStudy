import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, kernel_size=5, stride=1, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.liner1 = Linear(1024, 64)
        # self.liner2 = Linear(64, 10)

        self.model1 = torch.nn.Sequential(
            Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.liner1(x)
        # x = self.liner2(x)
        x = self.model1(x)
        return x

net = Net()
print(net)
input = torch.ones((64, 3, 32, 32))
output = net(input)
print(output.shape)

writer = SummaryWriter('./logs_seq')
writer.add_graph(net, input)

writer.close()
