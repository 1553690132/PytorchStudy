import json
import os
import tensorflow as tf
from matplotlib import pyplot as plt

from model import *

from keras.src.legacy.preprocessing.image import ImageDataGenerator

data_root = os.path.abspath(os.path.join(os.getcwd(), '../'))
image_path = data_root + '/torch/data_set/flower_data/'
train_path = image_path + 'train'
valid_path = image_path + 'val'

# 创建文件保存权重
if not os.path.exists("save_weights"):
    os.mkdir("save_weights")

im_height = 224
im_width = 224
batch_size = 32
epochs = 10

# 预处理
train_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
valid_image_generator = ImageDataGenerator(rescale=1. / 255)

# categorical代表使用one-hot编码，适用于多分类任务
train_data_generator = train_image_generator.flow_from_directory(directory=train_path,
                                                                 batch_size=batch_size,
                                                                 shuffle=True,
                                                                 target_size=(im_height, im_width),
                                                                 class_mode='categorical')
total_train = train_data_generator.n

test_data_generator = valid_image_generator.flow_from_directory(directory=valid_path,
                                                                batch_size=batch_size,
                                                                shuffle=False,
                                                                target_size=(im_height, im_width),
                                                                class_mode='categorical')

total_test = test_data_generator.n

# 反转字典便于后续对应
class_indices = train_data_generator.class_indices
inverse_dict = dict((val, key) for key, val in class_indices.items())

json_str = json.dumps(inverse_dict, indent=4)
with open("class_indices.json", "w") as json_file:
    json_file.write(json_str)

model = AlexNet_v1(im_height, im_width, 5)
model.summary()  # 打印神经网络具体信息

# 利用keras的api训练
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              # from_logits在设置softmax处理后设为false即可
              metrics=['accuracy'])

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/myAlex.weights.h5',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                monitor='val_loss')]

history = model.fit(x=train_data_generator,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=test_data_generator,
                    validation_steps=total_test // batch_size,
                    callbacks=callbacks)

history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.figure()
plt.plot(range(epochs), train_loss, label='Training loss')
plt.plot(range(epochs), val_loss, label='Validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.figure()
plt.plot(range(epochs), train_acc, label='train_accuracy')
plt.plot(range(epochs), val_acc, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
