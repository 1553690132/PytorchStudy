import json

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from model import *

im_height = 224
im_width = 224

img = Image.open('../torch/data_set/img.png')
img = img.resize((im_height, im_width))
plt.imshow(img)

# 转化图片为0-1间的点阵
img = np.array(img) / 255.0
# 给图片增加batch维度
img = np.expand_dims(img, axis=0)

try:
    json_file = open('./class_indices.json', 'r')
    class_indices = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

model = AlexNet_v1(class_num=5)
model.load_weights('./save_weights/myAlex.weights.h5')
result = np.squeeze(model.predict(img))
predict_class = np.argmax(result)
print(class_indices[str(predict_class)], result[predict_class])
plt.show()
