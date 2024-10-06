import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from model import *

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform
                                        , download=True)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=10000, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

epoch_num = 10

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

test_iter = iter(test_loader)
test_images, test_labels = test_iter.__next__()

# def imshow(img):
#     img = img / 2 + 0.5  # 反标准化: input = output * 0.5 + 0.5 = output / 2 + 0.5  标准化: output = (input - 0.5) / 0.5
#     np_img = img.numpy()
#     plt.imshow(np.transpose(np_img, (1, 2, 0)))
#     plt.show()
#
#
# # show images
# imshow(torchvision.utils.make_grid(test_images))
# # print labels
# print(' '.join(f'{classes[test_labels[j]]:5s}' for j in range(4)))

loss_val = 0

for epoch in range(epoch_num):
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val += loss.item()

        model.eval()
        if step % 500 == 499:
            with torch.no_grad():
                outputs = model(test_images)
                predicted = outputs.argmax(1)  # 获取index值
                accuracy = (predicted == test_labels).sum().item() / test_labels.size(0)

                print(f"train_loss: {loss_val / 500:.4f} accuracy: {accuracy:.4f}")
                loss_val = 0

save_path = './lenet.pth'
torch.save(model.state_dict(), save_path)
