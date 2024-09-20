from tensorflow.keras import Model, layers, Sequential


def VGG(feature, im_height, im_width, num_classes=1000):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype='float32')
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Softmax()(x)
    model = Model(inputs=input_image, outputs=outputs)
    return model


def make_feature(cfg):
    feature_layers = []
    for v in cfg:
        if v == 'M':
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            feature_layers.append(layers.Conv2D(filters=v, kernel_size=3, padding='same', activation='relu'))
    return Sequential(feature_layers, name="feature")


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", im_height=224, im_width=224, num_classes=1000):
    cfg = cfgs[model_name]
    model = VGG(make_feature(cfg), im_height=im_height, im_width=im_width, num_classes=num_classes)
    return model
