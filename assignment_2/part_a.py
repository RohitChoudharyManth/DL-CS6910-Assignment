
import os
import numpy as np
import tensorflow as tf
import keras
import wandb
from tensorflow.keras import layers,models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Flatten, Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator


import wandb
from wandb.keras import WandbCallback

wandb.login()

train_path='./nature_12K/inaturalist_12K/train'
test_path='./nature_12K/inaturalist_12K/val'
labels=['Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia']



def Train_CNN():
    image_size = (256, 256)
    default_configs = {"epochs": 15 ,
                       'activation': 'relu',
                       'filter_size': (3, 3),
                       'filter_1': 32,
                       'filter_2': 32,
                       'filter_3': 128,
                       'filter_4': 128,
                       'filter_5': 128,
                       'filter_organization': 'same',
                       'data_augmentation': 'no',
                       'dense_neuron_count': 32,
                       'num_classes': 10,
                       'image_size': (128, 128),
                       'batch_size': 64,
                       'dropout': 0.1,
                       'batch_norm': True
                       }

    run = wandb.init(project='cs6910-assignment2', config=default_configs)
    config = wandb.config
    dropout = config.dropout
    activation_function = config.activation
    filter_size = config.filter_size
    batch_norm_flag = config.batch_norm
    data_augmentation = config.data_augmentation
    dense_Neuron_count = config.dense_neuron_count

    if config.data_augmentation == "yes":
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


    validation_gen = train_data.flow_from_directory(
        train_path,
        target_size=image_size,  # resize to this size
        subset='validation',
        color_mode='rgb',  # for coloured images
        batch_size=config.batch_size,  # number of images to extract from folder for every batch
        # Size of the batches of data (default: 32)
        class_mode='sparse')  # classes to predict, "sparse" will be 1D integer labels


    test_generator = test_data.flow_from_directory(
        test_path,
        target_size=image_size,  # resize to this size
        batch_size=config.batch_size,  # number of images to extract from folder for every batch
        # Size of the batches of data (default: 32)
        class_mode='sparse')

    model = models.Sequential()
    if batch_norm_flag:
        model.add(BatchNormalization(input_shape=(image_size[0], image_size[1], 3)))
        model.add(Conv2D(config.filter_1, filter_size))
    else:
        model.add(Conv2D(config.filter_1, filter_size, input_shape=(image_size[0], image_size[1], 3)))
    # CNN_Layer_1
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    # CNN_Layer_2
    model.add(Conv2D(config.filter_2, filter_size))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


    # CNN_Layer_3
    model.add(Conv2D(config.filter_3, filter_size))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


    # CNN_Layer_4
    model.add(Conv2D(config.filter_4, filter_size))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    # CNN_Layer_5
    model.add(Conv2D(config.filter_5, filter_size))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(dense_Neuron_count, Activation(activation_function)))
    model.add(Dense(10, Activation('softmax')))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=[tf.keras.losses.SparseCategoricalCrossentropy()], metrics=['accuracy'])
    model.summary()
    H = model.fit(train_gen, epochs=config.epochs, validation_data=validation_gen)
    model.save('best_model.h5')
    val_acc = 100*max(H.history['val_accuracy'])



    params = {'activation': activation_function, 'filter_size': filter_size, 'filter_layer_1': config.filter_1,
              'filter_layer_2': config.filter_2, 'filter_layer_3': config.filter_3, 'filter_layer_4': config.filter_4,
              'filter_layer_5': config.filter_5, 'augmentation': data_augmentation, 'val_acc': val_acc,
              'dense_neuron_count': dense_Neuron_count, 'batch_norm': batch_norm_flag, 'dropout': dropout, 'section': 'Test_Best'}
    wandb.log(params)

Train_CNN()



# sweep_config = {
# 'method': 'bayes', #grid, random
# 'metric': {
#     'name': 'accuracy',
#     'goal': 'maximize'
#  },
# 'parameters': {
# 'activation' : {
#     'values': ['relu']
#     },
# 'dense_neuron_count' : {
#     'values': [16]
#     },
# 'filter_1' : {
#     'values': [32]
#     },
# 'filter_2' : {
#     'values': [32]
#     },
# 'filter_3' : {
#     'values': [128]
#     },
# 'filter_4' : {
#     'values': [128]
#     },
# 'filter_5' : {
#     'values': [128]
#     },
#     'data_augmentation' : {
#     'values': ['no']
#     },
#     'dropout': {
#         'values': [0.1]
#     },
#     'batch_norm': {
#         'values': [True]
#     }
#
# }
# }
# #
# sweep_id = wandb.sweep(sweep_config, project="cs6910-assignment2")
# wandb.agent(sweep_id, Train_CNN,count = 100)




def return_filter_values(filter_organization='normal', base_filter_val=64):
    if filter_organization=='normal':
        return [base_filter_val]*5
    elif filter_organization=='double':
        return [i*base_filter_val for i in range(1, 6)]
    elif filter_organization=='half':
        return [base_filter_val/i for i in range(1, 6)]




    # # CNN_Layer_1
    # model.add(Activation(activation_function))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(dropout))
    # # CNN_Layer_2
    # model.add(Conv2D(config.filter_2, filter_size))
    # model.add(Activation(activation_function))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(dropout))
    #
    #
    # # CNN_Layer_3
    # model.add(Conv2D(config.filter_3, filter_size))
    # model.add(Activation(activation_function))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(dropout))
    #
    #
    # # CNN_Layer_4
    # model.add(Conv2D(config.filter_4, filter_size))
    # model.add(Activation(activation_function))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(dropout))
    #
    # # CNN_Layer_5
    # model.add(Conv2D(config.filter_5, filter_size))
    # model.add(Activation(activation_function))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(dropout))