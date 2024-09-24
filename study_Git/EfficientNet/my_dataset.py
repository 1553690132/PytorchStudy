import torch
from PIL import Image
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[idx]))
        label = self.images_class[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 解包批次数据 (batch 是一个元组列表，包含图片和标签)
        images, labels = tuple(zip(*batch))  # 将图片和标签分别提取出来
        # 将多个图片张量拼接成一个批次张量
        images = torch.stack(images, 0)  # 将图片拼接成一个形状为 (batch_size, C, H, W) 的张量
        # 将标签转换为张量
        labels = torch.as_tensor(labels)  # 将标签转为张量 (batch_size,)
        return images, labels
