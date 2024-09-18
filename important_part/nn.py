# import torch
# import torchvision
# import torch.nn
# from torch.utils.data import DataLoader
#
# from torch.utils.tensorboard import SummaryWriter
#
# writer = SummaryWriter('logs')
#
# dataset = torchvision.datasets.CIFAR10(root='./dataset', transform=torchvision.transforms.ToTensor(), dog=False,
#                                        download=True)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
#
#
# class Test(torch.nn.Module):
#     def __init__(self):
#         super(Test, self).__init__()
#         self.conv1 = torch.nn.Conv2d(3, 6, 3, padding=0, stride=1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
#
#
# test = Test()
# for index, data in enumerate(dataloader):
#     imgs, targets = data
#     writer.add_images('normal', imgs, index)
#
#     conv_imgs = torch.reshape(test(imgs), (-1, 3, 30, 30))
#     writer.add_images('conv2d', conv_imgs, index)
#
#
# writer.close()

import torch
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


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
            Linear(1024,64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

net = Net()
print(net)

writer = SummaryWriter('./logss')
input_v = torch.ones((64, 3, 32, 32))
output_v = net(input_v)
print(output_v.shape)

writer.add_graph(net, input_v)

writer.close()
