import torch
import torchvision

from torch import nn

# 方式1
vgg16 = torchvision.models.vgg16(pretrained=True)

torch.save(vgg16, 'vgg16_method1.pth')

# 2 (推荐) 按字典保存
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')


# 陷阱
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)

    def forward(self, x):
        return self.conv1(x)


net = Net()
torch.save(net, 'net.pth')
