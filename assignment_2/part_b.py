import os
import numpy as np
import tensorflow as tf
import keras
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D

import wandb
from tensorflow.keras import layers,models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Flatten, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
#from keras_applications.resnet50 import ResNet50
#from keras.applications.xception import Xception

train_path='./nature_12K/inaturalist_12K/train'
test_path='./nature_12K/inaturalist_12K/val'
labels=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']


def lr_scheduler(epoch, lr):
    decay_rate = 0.5
    decay_step = 4
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


def Pretraining():
    image_size = (224, 224)
    default_configs = {'activation': 'relu',
                       'data_augmentation': 'yes',
                       'batch_size': 64,
                        'num_layers_unfreeze_base' : 0
                       }

    run = wandb.init(project='cs6910-assignment2', config=default_configs)
    config = wandb.config
    num_layers_unfreeze = config.num_layers_unfreeze_base
    activation_function = config.activation
    data_augmentation = config.data_augmentation


    if data_augmentation == "yes":
        train_data = ImageDataGenerator(rescale=1. / 255,
                                        validation_split=0.1,
                                        rotation_range=90,  # rotation angle in degrees
                                        fill_mode='nearest',  #
                                        width_shift_range=0.2,  # Shift image
                                        height_shift_range=0.2,
                                        zoom_range=0.2,  # value smaller than 1 will zoom in
                                        shear_range=0.2,
                                        horizontal_flip=True,  # Flip images
                                        vertical_flip=True)

    else:
        train_data = ImageDataGenerator(rescale=1. / 255, validation_split=0.1)

    test_data = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_data.flow_from_directory(
        train_path,
        target_size=image_size,  # resize to this size
        subset='training',
        color_mode='rgb',  # for coloured images
        batch_size=config.batch_size,
        # number of images to extract from folder for every batch # Size of the batches of data (default: 32)
        class_mode='sparse')  # classes to predict, "sparse" will be 1D integer labels
    # shuffle = True) # Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.

    # seed=2020 # to make the result reproducible
    # )

    validation_gen = train_data.flow_from_directory(
        train_path,
        target_size=image_size,  # resize to this size
        subset='validation',
        color_mode='rgb',  # for coloured images
        batch_size=config.batch_size,  # number of images to extract from folder for every batch
        # Size of the batches of data (default: 32)
        class_mode='sparse')  # classes to predict, "sparse" will be 1D integer labels
    # shuffle = True) # Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.

    # seed=2020 # to make the result reproducible
    # )

    test_generator = test_data.flow_from_directory(
        test_path,
        target_size=image_size,  # resize to this size
        batch_size=config.batch_size,  # number of images to extract from folder for every batch
        # Size of the batches of data (default: 32)
        class_mode='sparse')  # classes to predict, "sparse" will be 1D integer labels

    img_height = 224
    img_width = 224

    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))
    base_model.summary()
    if num_layers_unfreeze > 0:
        for layers in base_model.layers[:-num_layers_unfreeze]:
            layers.trainable = False
    else:
        for layers in base_model.layers:
            layers.trainable = False

    model = keras.Sequential([tf.keras.Input(shape=(img_height, img_width, 3,)), base_model, GlobalAveragePooling2D(), Flatten(),
                              Dense(1024, activation=activation_function), Dense(512, activation=activation_function),
                              Dense(64, activation=activation_function), Dense(10, activation='softmax')])

    # tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
        metrics=['accuracy'],
    )
    epochs = 15
    H = model.fit(train_gen, epochs=epochs, validation_data=validation_gen)
    #
    val_acc = max(H.history['val_accuracy'])

    params = {'augmentation': data_augmentation, 'val_acc': val_acc,
              'section': 'B_inceptionv2', 'number_of_layers_unfreezed': num_layers_unfreeze, 'epochs': epochs, 'pooling':'max'}

    wandb.log(params)
Pretraining()

# sweep_config = {
#     'name': 'sweep',
#     'method': 'random',
#     'parameters': {
#         'data_augmentation': {
#             'values': ['yes', 'no']
#         },
#         'num_layers_unfreeze_base': {
#             'values': [0, 1, 3]
#         },
#     }
#
# }
# sweep_id = wandb.sweep(sweep_config, project="cs6910-assignment2")
# wandb.agent(sweep_id, Pretraining,count = 100, project="cs6910-assignment2")


# wandb.agent('2k00o9ov', Pretraining,count = 100, project="cs6910-assignment2")