import json
import os

from matplotlib import pyplot as plt

from model import *

from keras.src.legacy.preprocessing.image import ImageDataGenerator
import tensorflow as tf

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = os.path.join(data_root, "data_set", "flower_data")
train_dir = os.path.join(image_path, "train")
validation_dir = os.path.join(image_path, "val")

if not os.path.exists('save_weights'):
    os.mkdir('save_weights')

im_height = 224
im_width = 224
batch_size = 32
epochs = 10

train_image_generator = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True)
validation_image_generator = ImageDataGenerator(rescale=1. / 255)

train_data_gen = train_image_generator.flow_from_directory(directory=train_dir,
                                                           batch_size=batch_size,
                                                           shuffle=True,
                                                           target_size=(im_height, im_width),
                                                           class_mode='categorical')
validation_data_gen = validation_image_generator.flow_from_directory(directory=validation_dir,
                                                                     batch_size=batch_size,
                                                                     shuffle=False,
                                                                     target_size=(im_height, im_width),
                                                                     class_mode='categorical')
total_train = train_data_gen.n
total_validation = validation_data_gen.n

class_names = train_data_gen.class_indices
inverse_dict = dict((val, key) for key, val in class_names.items())

json_str = json.dumps(inverse_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

model = vgg("vgg16", im_height, im_width, num_classes=5)
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath='./save_weights/vgg16.weights.h5',
                                                save_best_only=True,
                                                save_weights_only=True,
                                                monitor='val_loss', )]
history = model.fit(x=train_data_gen,
                    steps_per_epoch=total_train // batch_size,
                    epochs=epochs,
                    validation_data=validation_data_gen,
                    validation_steps=total_validation // batch_size,
                    callbacks=callbacks)

history_dict = history.history
train_loss = history_dict['loss']
val_loss = history_dict['val_loss']
train_acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.figure()
plt.plot(range(epochs), train_loss, label='train_loss')
plt.plot(range(epochs), val_loss, label='val_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')

# figure 2
plt.figure()
plt.plot(range(epochs), train_acc, label='train_accuracy')
plt.plot(range(epochs), val_acc, label='val_accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
