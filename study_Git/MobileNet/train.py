import json
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model_v2 import *
from model_v3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
epochs = 5

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transform["train"])
val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'), transform=data_transform["val"])

num_train = len(train_dataset)
num_val = len(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cla_dict, indent=4)
with open("class_indices.json", "w") as writer:
    writer.write(json_str)

net = mobilenet_v3_large(num_classes=5)
pre_weights = torch.load('./mobilenet_v3_large-8738ca79.pth')


pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}

missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

# 冻结除最后层的权重
for param in net.features.parameters():
    param.requires_grad = False

net.to(device)

loss_fn = nn.CrossEntropyLoss()

params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

best_acc = 0
save_path = './MobileNetV3.pth'
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs.to(device))
        loss = loss_fn(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    net.eval()
    acc = 0.0
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for step, data in enumerate(val_bar):
            inputs, labels = data
            outputs = net(inputs.to(device))
            predict = torch.argmax(outputs, dim=1)
            acc += torch.eq(predict, labels.to(device)).sum().item()

            val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                       epochs)
    val_acc = acc / num_val
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / len(train_loader), val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), save_path)

print('Finished Training')
