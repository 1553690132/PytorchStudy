import torch
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, MaxPool2d, Conv2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = Sequential(
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
        x = self.model(x)
        return x


net = Net()
print(net)

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.01)

for i in range(10):
    running_loss = 0.0
    for index, data in enumerate(dataloader):
        imgs, targets = data
        outputs = net(imgs)
        result = loss(outputs, targets)
        optim.zero_grad()
        result.backward()
        optim.step()
        running_loss = running_loss + result
    print(running_loss)
