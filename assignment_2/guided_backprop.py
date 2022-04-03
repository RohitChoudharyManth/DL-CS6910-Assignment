
import os
import numpy as np
import tensorflow as tf
import keras
import wandb
from tensorflow.keras import layers,models
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization, Flatten, Dropout
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

import wandb
from wandb.keras import WandbCallback


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
    dropout = 0.1
    activation_function = 'relu'
    batch_norm_flag = 'yes'
    data_augmentation = 'no'
    dense_Neuron_count = 32
    filter_size = (3, 3)
    batch_size = 64
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
        batch_size=batch_size,
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
        batch_size=batch_size,  # number of images to extract from folder for every batch
        # Size of the batches of data (default: 32)
        class_mode='sparse')  # classes to predict, "sparse" will be 1D integer labels
    # shuffle = True) # Whether to shuffle the data (default: True) If set to False, sorts the data in alphanumeric order.

    # seed=2020 # to make the result reproducible
    # )

    test_generator = test_data.flow_from_directory(
        test_path,
        target_size=image_size,  # resize to this size
        batch_size=batch_size,  # number of images to extract from folder for every batch
        # Size of the batches of data (default: 32)
        class_mode='sparse')  # classes to predict, "sparse" will be 1D integer labels

    model = models.Sequential()
    if batch_norm_flag:
        model.add(BatchNormalization(input_shape=(image_size[0], image_size[1], 3)))
        model.add(Conv2D(32, filter_size))
    else:
        model.add(Conv2D(config.filter_1, filter_size, input_shape=(image_size[0], image_size[1], 3)))
    # CNN_Layer_1
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    # CNN_Layer_2
    model.add(Conv2D(32, filter_size))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


    # CNN_Layer_3
    model.add(Conv2D(128, filter_size))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))


    # CNN_Layer_4
    model.add(Conv2D(128, filter_size))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    # CNN_Layer_5
    model.add(Conv2D(128, filter_size))
    model.add(Activation(activation_function))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(dense_Neuron_count, Activation(activation_function)))
    model.add(Dense(10, Activation('softmax')))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=[tf.keras.losses.SparseCategoricalCrossentropy()], metrics=['accuracy'])
    model.summary()

    # H = model.fit(train_gen, epochs=10, validation_data=validation_gen)
    # model.save('best_model.h5')
    # val_acc = 100*max(H.history['val_accuracy'])

    # summarize filter shapes
    for layer in model.layers:
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue



    # get filter weights
    LAYER_NAME = 'conv2d'
    # filters, biases = model.get_layer(LAYER_NAME).get_weights()
    # print(layer.name, filters.shape)
    #
    # # normalize filter values to 0-1 so we can visualize them
    # f_min, f_max = filters.min(), filters.max()
    # filters = (filters - f_min) / (f_max - f_min)
    #
    # n_filters, ix = 32, 1
    # for i in range(n_filters):
    #     # get the filter
    #     f = filters[:, :, :, i]
    #     # plot each channel separately
    #     for j in range(3):
    #         # specify subplot and turn of axis
    #         ax = plt.subplot(8, 3*4, ix)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         # plot filter channel in grayscale
    #         plt.imshow(f[:, :, j], cmap='gray')
    #         ix += 1
    # # show the figure
    # plt.show()

    gb_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(LAYER_NAME).output]
    )
    layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer, 'activation')]

    @tf.custom_gradient
    def guidedRelu(x):
        def grad(dy):
            return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

        return tf.nn.relu(x), grad


    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guidedRelu

    for x, y in test_generator:
        img = x[0]
        print(x.shape)
        break
    img = np.expand_dims(img, axis=0)
    print(img.shape)
    with tf.GradientTape() as tape:
        inputs = tf.cast(img, tf.float32)
        tape.watch(inputs)
        outputs = gb_model(inputs)[0]
    grads = tape.gradient(outputs, inputs)[0]

    guided_back_prop = grads
    gb_viz = np.dstack((
        guided_back_prop[:, :, 0],
        guided_back_prop[:, :, 1],
        guided_back_prop[:, :, 2],
    ))
    gb_viz -= np.min(gb_viz)
    gb_viz /= gb_viz.max()

    imgplot = plt.imshow(gb_viz)
    plt.axis("off")
    plt.show()

Train_CNN()



# gb_model = tf.keras.models.Model(
#     inputs=[model.inputs],
#     outputs=[model.get_layer(LAYER_NAME).output]
# )
# layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer, 'activation')]


# @tf.custom_gradient
# def guidedRelu(x):
#     def grad(dy):
#         return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
#
#     return tf.nn.relu(x), grad
#
#
# for layer in layer_dict:
#     if layer.activation == tf.keras.activations.relu:
#         layer.activation = guidedRelu
#
# for x, y in train_gen:
#     img = x[0]
#     print(x.shape)
#     break
# img = np.expand_dims(img, axis=0)
# print(img.shape)
# with tf.GradientTape() as tape:
#     inputs = tf.cast(img, tf.float32)
#     tape.watch(inputs)
#     outputs = gb_model(inputs)[0]
# grads = tape.gradient(outputs, inputs)[0]
# weights = tf.reduce_mean(grads, axis=(0, 1))
# grad_cam = np.ones(outputs.shape[0: 2], dtype=np.float32)
# for i, w in enumerate(weights):
#     grad_cam += w * outputs[:, :, i]
#
# grad_cam_img = cv2.resize(grad_cam.numpy(), (256, 256))
# grad_cam_img = np.maximum(grad_cam_img, 0)
# heatmap = (grad_cam_img - grad_cam_img.min()) / (grad_cam_img.max() - grad_cam_img.min())
# grad_cam_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
# print(grad_cam_img.shape)
# print(img.shape)
# output_image = cv2.addWeighted(cv2.cvtColor(np.squeeze(img).astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, grad_cam_img, 1,
#                                0)
#
# plt.imshow(output_image)
# plt.axis("off")
# plt.show()
#
