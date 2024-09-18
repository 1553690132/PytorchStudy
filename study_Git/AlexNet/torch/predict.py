import json

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from model import *

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img = Image.open("./data_set/img.png")
plt.imshow(img)
img = data_transform(img)
# 添加batch参数
img = torch.unsqueeze(img, 0)

try:
    json_file = open('class_dict.json', 'r')
    class_dict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = AlexNet(num_classes=5)
model.load_state_dict(torch.load("./AlexNet_model.pth"))
model.eval()

with torch.no_grad():
    # 压缩掉batch参数
    output = torch.squeeze(model(img))
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

print(class_dict[str(predict_cla)], predict[predict_cla].item())
plt.show()

