import torch
import torchvision

# 方式1
model = torch.load('./vgg16_method1.pth')
print(model)

# 2
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load('./vgg16_method2.pth'))
print('====\n', vgg16)

# 陷阱
model2 = torch.load('./net.pth')
print(model2)
