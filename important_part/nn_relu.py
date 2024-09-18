import torch
import torchvision.datasets
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

output = torch.reshape(input, (-1, 1, 2, 2))

writer = SummaryWriter('./logs_relu')


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        output = self.sigmoid1(x)
        return output


# net = Net()
# output = net(output)
# print(output)

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

net2 = Net()
for index, data in enumerate(dataloader):
    imgs, targets = data
    writer.add_images('input', imgs, index)
    writer.add_images('sigomid', net2(imgs), index)

writer.close()
