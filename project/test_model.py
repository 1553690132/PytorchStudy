import torch
import torchvision
from PIL import Image
from torch.nn import MaxPool2d, Conv2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2, stride=1),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


writer = SummaryWriter(log_dir='./logs')
net = Net()
net.load_state_dict(torch.load('./99.pth'))

img_path = './img/img_6.png'
img = Image.open(img_path)
transforms = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

img_dataset = torchvision.datasets.ImageFolder(root='./test_data/train', transform=transforms)

img_dataloader = DataLoader(img_dataset, batch_size=3, shuffle=True)

net.eval()

for data in img_dataloader:
    imgs, labels = data
    with torch.no_grad():
        outs = net(imgs)

    print(outs.argmax(dim=1))


img1_path = './test_data/train/cat/img.png'
img1 = Image.open(img1_path)

img1 = transforms(img1)
img1 = torch.reshape(img1, (1, 3, 32, 32))

net.eval()
with torch.no_grad():
    outs = net(img1)
print(outs.argmax(dim=1))

# img = transforms(img)
#
# print(img.shape)
#
#
# img = torch.reshape(img, (1, 3, 32, 32))
# net.eval()
#
# with torch.no_grad():
#     out = net(img)
#
# print(out.argmax(dim=1))
