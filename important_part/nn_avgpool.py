import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.avgpool1 = nn.AvgPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.avgpool1(x)
        return x


writer = SummaryWriter('./logs_avgpool')

net = Net()

for index, data in enumerate(dataloader):
    imgs, targets = data
    writer.add_images('imgs', imgs, index)
    avg_imgs = net(imgs)
    writer.add_images('avgpool', avg_imgs, index)

writer.close()
