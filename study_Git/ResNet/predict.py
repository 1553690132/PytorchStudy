import json

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img_path = '../data_set/img.png'
img = Image.open(img_path)
plt.imshow(img)

img = data_transform(img)
img = torch.unsqueeze(img, 0)

json_path = 'class_indices.json'
with open(json_path, 'r') as f:
    class_names = json.load(f)

model = resnet34(5).to(device)
model.load_state_dict(torch.load("./resnet34.pth", map_location=device))

model.eval()
with torch.no_grad():
    output = torch.squeeze(model(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict, dim=0).numpy()

print_res = "class: {}   prob: {:.3}".format(class_names[str(predict_cla)],
                                             predict[predict_cla].numpy())
plt.title(print_res)
for i in range(len(predict)):
    print("class: {:10}   prob: {:.3}".format(class_names[str(i)],
                                              predict[i].numpy()))
plt.show()
