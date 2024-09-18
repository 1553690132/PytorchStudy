import torch
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# input_v = torch.tensor([[1, 2, 0, 3, 1],
#                         [0, 1, 2, 3, 1],
#                         [1, 2, 1, 0, 0],
#                         [5, 2, 3, 1, 1],
#                         [2, 1, 0, 1, 1]], dtype=torch.float32)
#
# input_v = torch.reshape(input_v, (-1, 1, 5, 5))

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        x = self.maxpool1(x)
        return x


net = Net()
writer = SummaryWriter('./logs_maxpool')

for index, data in enumerate(dataloader):
    imgs, targets = data
    writer.add_images('input', imgs, index)
    outputs = net(imgs)
    writer.add_images('maxpool', outputs, index)

writer.close()
