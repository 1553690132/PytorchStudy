import torch
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    # 使宽高减半
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)


# 每个卷积层的层数和输入输出通道数
conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512),
             (2, 512, 512))

fc_feature = 512 * 7 * 7
fc_hidden_units = 4096


def vgg_11(conv_arch, fc_feature, fc_hidden_units=4096):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module(f'vgg_block_{i}', vgg_block(num_convs, in_channels, out_channels))
        vgg_block(num_convs, in_channels, out_channels)
    net.add_module('fc', nn.Sequential(nn.Flatten(),
                                       nn.Linear(fc_feature, fc_hidden_units),
                                       nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(), nn.Dropout(0.5), nn.Linear(fc_hidden_units, 10)))
    return net


net = vgg_11(conv_arch, fc_feature, fc_hidden_units)
x = torch.rand(1, 1, 224, 224)

for name, blk in net.named_children():
    x = blk(x)
    print(name, 'output shape', x.shape)

print(net)
