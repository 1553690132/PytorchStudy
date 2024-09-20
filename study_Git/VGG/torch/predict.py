import json

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = Image.open('../../data_set/img.png')
plt.imshow(img)
transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

img = transform(img)
img = torch.unsqueeze(img, 0)

with open('./class_indices.json', 'r') as f:
    class_names = json.load(f)

model = vgg('vgg16', num_classes=5).to(device)
model.load_state_dict(torch.load('./vgg16Net.pth', map_location=device))

model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

print(class_names[str(predict_cla)], predict[predict_cla].item())
plt.show()
