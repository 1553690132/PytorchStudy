import torch
import torchvision
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

# 加载数据集
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# 创建网络模型
net = Net()

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()

# 优化器
learn_rate = 1e-2
optimizer = torch.optim.SGD(net.parameters(), learn_rate)

# 设置训练网络参数
# 训练次数
total_train_step = 0
# 测试次数
total_test_step = 0
# 训练轮数
epoch = 10

writer = SummaryWriter('./logs')

import time

for i in range(epoch):
    print(f"第{i}次训练开始！")
    start_time = time.time()

    net.train()
    for data in train_loader:
        images, targets = data
        output_v = net(images)
        loss = loss_fn(output_v, targets)

        # 优化器调优模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print(f"训练次数：{total_train_step} 损失值：{loss.item()}")
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    net.eval()
    # 测试
    total_test_loss = 0
    # 整体正确值
    total_accuracy = 0

    with torch.no_grad():
        for data in test_loader:
            images, targets = data
            output_v = net(images)
            loss = loss_fn(output_v, targets)
            total_test_loss = total_test_loss + loss.item()

            accuracy = (output_v.argmax(dim=1).eq(targets)).sum()
            total_accuracy += accuracy.item()

    print(f"整体测试集上的loss：{total_test_loss}")
    print(f"整体测试集上的正确率：{total_accuracy / test_data_size}")
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(net.state_dict(), f'./{i}.pth')

writer.close()
