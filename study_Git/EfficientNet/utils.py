import json
import os
import pickle
import random
import sys

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可以复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 遍历文件夹 找出类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()))
    with open('class_indices.json', 'w') as f:
        f.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        # 记录类别数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

        print("{} images were found in the dataset.".format(sum(every_class_num)))
        print("{} images for training.".format(len(train_images_path)))
        print("{} images for validation.".format(len(val_images_path)))
        assert len(train_images_path) > 0, "number of training images must greater than 0."
        assert len(val_images_path) > 0, "number of validation images must greater than 0."

        plot_image = False
        if plot_image:
            # 绘制每种类别个数柱状图
            plt.bar(range(len(flower_class)), every_class_num, align='center')
            # 将横坐标0,1,2,3,4替换为相应的类别名称
            plt.xticks(range(len(flower_class)), flower_class)
            # 在柱状图上添加数值标签
            for i, v in enumerate(every_class_num):
                plt.text(x=i, y=v + 5, s=str(v), ha='center')
            # 设置x坐标
            plt.xlabel('image class')
            # 设置y坐标
            plt.ylabel('number of images')
            # 设置柱状图的标题
            plt.title('flower class distribution')
            plt.show()

        return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        loss = loss_fn(pred, labels.to(device))
        loss.backward()
        # 计算到当前 step 为止的平均损失，并使用 `detach` 来避免对图计算图的影响
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)

        data_loader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    total_num = len(data_loader.dataset)
    # 存储正确样本个数
    sum_num = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        output = model(images.to(device))
        pred = torch.argmax(output, dim=1)
        sum_num += torch.eq(pred, labels.to(device)).sum().item()

    return sum_num / total_num
