import json
import os.path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

data_root = os.path.abspath(os.path.join(os.getcwd(), '../..'))
img_path = os.path.join(data_root, 'data_set', 'flower_data')

train_dataset = datasets.ImageFolder(root=os.path.join(img_path, 'train'), transform=data_transform['train'])
val_dataset = datasets.ImageFolder(root=os.path.join(img_path, 'val'), transform=data_transform['val'])

num_train = len(train_dataset)
num_val = len(val_dataset)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())

json_str = json.dumps(cla_dict)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
epochs = 30

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model_name = 'vgg16'
model = vgg(model_name, num_classes=5, init_weights=True)
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

best_acc = 0.0
save_path = "./{}Net.pth".format(model_name)

print(len(train_dataloader), len(train_dataset))

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_bar = tqdm(train_dataloader)
    for step, data in enumerate(train_bar):
        images, labels = data
        outputs = model(images.to(device))
        loss = loss_fn(outputs, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    model.eval()
    acc = 0.0
    with torch.no_grad():
        var_bar = tqdm(val_dataloader)
        for step, data in enumerate(var_bar):
            images, labels = data
            outputs = model(images.to(device))
            pred = outputs.argmax(1)
            acc += torch.eq(pred, labels.to(device)).sum().item()

    val_acc = acc / num_val
    print(f"epoch: {epoch + 1}, val_acc: {val_acc}, train_loss: {train_loss / len(train_dataloader)}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f"save model to {save_path}")
        print("best acc:", best_acc)

print('Finished Training')
