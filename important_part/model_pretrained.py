import torchvision
from torch.nn import Linear

vgg_16_false = torchvision.models.vgg16(pretrained=False)
vgg_16_true = torchvision.models.vgg16(pretrained=True)

print(vgg_16_false)

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg_16_true.classifier.add_module('add_liner', Linear(1000, 10))

print(vgg_16_true)

vgg_16_false.classifier[6] = Linear(4096, 10)
