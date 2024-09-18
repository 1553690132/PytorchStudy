import torch.nn
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)


class LWH(torch.nn.Module):
    def __init__(self):
        super(LWH, self).__init__()
        self.conv1 = Conv2d(3, 6, kernel_size=3, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return x


lwh = LWH()
print(lwh)

writer = SummaryWriter('../logs')

step = 0
for data in dataloader:
    imgs, targets = data
    output = lwh(imgs)
    writer.add_images('input', imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('ouptut', output, step)
    step += 1

writer.close()
