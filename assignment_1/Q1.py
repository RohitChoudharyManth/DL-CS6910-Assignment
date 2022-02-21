#!pip install wandb

print("Importing Packages....")
from keras.datasets import fashion_mnist
import numpy as np
import wandb
import matplotlib.pyplot as plt
import random

print("Done")

print("Loading Fashion_Mnist Dataset....")
#Use the standard train/test split of fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Done!!")

class_labels={0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"}

wandb.init(project='CS6910_aditya_rohit')

image_list = []
label_list = []

plt.figure(figsize=[15, 5])
for i in range(0,10):
  index_list = np.where(y_train == i)
  index_list_len = index_list[0].size
  index = index_list[0][random.randint(0,index_list_len)]
  image = x_train[index,:,:]
  plt.subplot(2, 5, i+1)
  plt.imshow(image)
  plt.title(class_labels[i])
  image_list.append(image)
  label_list.append(class_labels[i])

wandb.log({"Question 1": [wandb.Image(img, caption=caption) for img, caption in zip(image_list, label_list)]})
