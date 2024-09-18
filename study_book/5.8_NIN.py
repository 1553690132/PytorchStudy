import torch
from torch import nn


def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
    )
    return blk


net = nn.Sequential(
    nin_block(1, 96, 11, 4, 0),
    nn.MaxPool2d(3, 2),
    nin_block(96, 256, 5, 1, 2),
    nn.MaxPool2d(3, 2),
    nin_block(256, 384, 3, 1, 1),
    nn.MaxPool2d(3, 2),
    nn.Dropout(0.5),
    # 标签类别数为十
    nin_block(384, 10, 3, 1, 1),
    nn.AvgPool2d(5),

    # 四维转化为二维输出 （batch_size, 10）
    nn.Flatten(),
)

x = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    x = blk(x)
    print(name, 'output shape:', x.shape)
