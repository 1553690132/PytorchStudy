import json
import os

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
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
with open('class_induces.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

net = GoogLeNet(num_classes=5, aux_logits=True, init_weights=True)
net.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0003)

epochs = 30
best_acc = 0.0
save_path = './googlNet.pth'

for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        logits, aux_logits2, aux_logits1 = net(images.to(device))
        loss0 = loss_fn(logits, labels.to(device))
        loss1 = loss_fn(aux_logits1, labels.to(device))
        loss2 = loss_fn(aux_logits2, labels.to(device))
        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
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
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            pred = outputs.argmax(dim=1)
            acc += torch.eq(pred, val_labels.to(device)).sum().item()
    val_acc = acc / val_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / len(train_loader), val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(net.state_dict(), save_path)

print('Finished Training')
