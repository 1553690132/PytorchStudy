import os

from torchvision import datasets, transforms

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
image_path = data_root + '/data_set/flower_data/flower_photos/'

print(image_path)

train_dataset = datasets.ImageFolder(root=image_path + "train", transform=data_transforms['train'])
test_dataset = datasets.ImageFolder(root=image_path + "val", transform=data_transforms['test'])

train_num = len(train_dataset)

flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
