import torchvision

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../logs')

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', transform=dataset_transform, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', transform=dataset_transform, train=False, download=True)

for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()
