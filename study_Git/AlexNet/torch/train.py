import json
import os
import sys

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
from tqdm import tqdm

from model import *

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), './'))
image_path = data_root + '/data_set/flower_data/'

print(image_path)

train_dataset = datasets.ImageFolder(root=image_path + "train", transform=data_transforms['train'])
test_dataset = datasets.ImageFolder(root=image_path + "val", transform=data_transforms['test'])

train_num = len(train_dataset)
test_num = len(test_dataset)

flower_list = train_dataset.class_to_idx
# 索引字典进行反转，val->key
cla_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cla_dict, indent=4)
with open('class_dict.json', 'w') as json_file:
    json_file.writelines(json_str)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

net = AlexNet(num_classes=5, init_weights=True)
net = net.to(device)

loss_fnc = nn.CrossEntropyLoss()
loss_fnc = loss_fnc.to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0002)

save_path = 'AlexNet_model.pth'
best_accuracy = 0.0

for epoch in range(10):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_dataloader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))

        loss = loss_fnc(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 打印进度
        rate = (step + 1) / len(train_dataloader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {: ^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end='')
    print()

    net.eval()
    with torch.no_grad():
        accuracy = 0
        test_bar = tqdm(test_dataloader, file=sys.stdout)
        for test_images, test_labels in test_bar:
            outputs = net(test_images.to(device))
            _, predicted = torch.max(outputs, 1)
            accuracy += (predicted == test_labels.to(device)).sum().item()

        test_accuracy = accuracy / test_num
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(net.state_dict(), save_path)
        print('[epoch %d] train_loss: %.3f  test_Accuracy: %.3f' % (
            epoch + 1, running_loss / len(train_dataloader), test_accuracy))

print('Finished Training')
