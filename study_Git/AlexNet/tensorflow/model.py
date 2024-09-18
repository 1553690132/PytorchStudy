from keras import Model
from tensorflow.keras import models, layers


def AlexNet_v1(im_height=224, im_width=224, class_num=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')  # 224 224 3
    # 补padding，为了后续图像尺寸与AlexNet要求一致
    x = layers.ZeroPadding2D(((1, 2), (1, 2)))(input_image)  # 227 227 3
    x = layers.Conv2D(48, kernel_size=11, strides=4, padding='valid', activation='relu')(x)  # 55 55 48
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)  # 27 27 48
    x = layers.Conv2D(128, kernel_size=5, padding='same', activation='relu')(x)  # 27 27 128
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)  # 13 13 128
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)  # 13 13 192
    x = layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)  # 13 13 192
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')(x)  # 13 13 128
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)  # 6 6 128

    x = layers.Flatten()(x)  # 6*6*128=4068
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu')(x)  # 2048
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation='relu')(x)  # 2048
    x = layers.Dense(class_num)(x)

    predict = layers.Softmax()(x)
    model = models.Model(inputs=input_image, outputs=predict)
    return model


class AlexNet_v2(Model):
    def __init__(self, num_classes=1000):
        super(AlexNet_v2, self).__init__()
        self.features = models.Sequential([
            layers.ZeroPadding2D(((1, 2), (1, 2))),
            layers.Conv2D(48, kernel_size=11, strides=4, padding='valid', activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2),
            layers.Conv2D(128, kernel_size=5, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2),
            layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=3, strides=2),
            layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'), ]
        )

        self.flatten = layers.Flatten()
        self.classifier = models.Sequential([
            layers.Dropout(0.2),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes),
            layers.Softmax(), ]
        )

    def call(self, inputs):
        x = self.features(inputs)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
