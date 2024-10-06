import json

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from model import *

im_height = 224
im_width = 224
batch_size = 32
num_classes = 5

img_path = '../../data_set/img.png'
img = Image.open(img_path)

img = img.resize((im_height, im_width))
plt.imshow(img)

img = np.array(img) / 255.0
img = np.expand_dims(img, axis=0)

json_path = './class_indices.json'
with open(json_path, 'r') as f:
    class_names = json.load(f)

model = vgg("vgg16", im_height, im_width, num_classes)
weight_path = './save_weights/vgg16_weights.h5'
model.load_weights(weight_path)

result = np.squeeze(model.predict(img))
predict_class = np.argmax(result)

print_res = "class: {}   prob: {:.3}".format(class_names[str(predict_class)],
                                             result[predict_class])
plt.title(print_res)
for i in range(len(result)):
    print("class: {:10}   prob: {:.3}".format(class_names[str(i)],
                                              result[i]))
plt.show()
