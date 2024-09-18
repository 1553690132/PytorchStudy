import torch


def comp_conv2d(conv2d, x):
    x = x.view((1, 1) + x.shape)
    print(x.shape)
    y = conv2d(x)
    print(y.shape)
    return y.view(y.shape[2:])


conv2d = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1)
x = torch.rand(8, 8)
print(x.shape)
y = comp_conv2d(conv2d, x)
print(y.shape)

conv2d_2 = torch.nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
y2 = comp_conv2d(conv2d_2, x)
print(y2.shape)
