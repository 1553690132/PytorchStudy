import json

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from model_v2 import *

img = Image.open('../data_set/img.png')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

plt.imshow(img)
img = data_transform(img)
img = torch.unsqueeze(img, 0)
img = img.to(device)

with open('./class_indices.json', 'r') as f:
    class_indict = json.load(f)

model = MobileNetV2(num_classes=5).to(device)
model.load_state_dict(torch.load("./MobileNetV2.pth", map_location=device))
model.eval()

with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()
    print(predict_cla)

print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                             predict[predict_cla].numpy())
plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                              predict[i].numpy()))
plt.show()
