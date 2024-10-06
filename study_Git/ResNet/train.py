import json
import os.path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../"))  # get data root path
image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
print(image_path)
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'train'), transform=data_transform['train'])
train_num = len(train_dataset)

val_dataset = datasets.ImageFolder(root=os.path.join(image_path, 'val'), transform=data_transform['val'])
val_num = len(val_dataset)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json.json', 'w') as f:
    f.write(json_str)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

net = resnet34()
net.load_state_dict(torch.load('./resnet34-333f7ec4.pth', map_location='cpu'))

# 改变原模型的全连接层
in_channel = net.fc.in_features
net.fc = nn.Linear(in_features=in_channel, out_features=5)

net.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

epochs = 3
best_acc = 0.0
save_path = './resNet34.pth'
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for i, data in enumerate(train_bar):
        inputs, labels = data
        optimizer.zero_grad()
        logits = net(inputs.to(device))
        loss = loss_fn(logits, labels.to(device))
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
        for i, data in enumerate(val_bar):
            val_inputs, val_labels = data
            outputs = net(val_inputs.to(device))
            predict_y = torch.argmax(outputs, dim=1)
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            val_bar.desc = "val epoch[{}/{}]".format(epoch + 1, epochs)

    val_acc = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / len(train_loader), val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), save_path)

print('Finished Training')
