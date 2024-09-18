import torchvision

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../dataloader')

test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=False, num_workers=0, drop_last=False)


print(test_loader)
step = 0

for epoch in range(2):
    for data in test_loader:
        img, target = data
        writer.add_images(f'No.{epoch}', img, step)
        step += 1

writer.close()
