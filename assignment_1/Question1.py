import numpy as np
from matplotlib import pyplot
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import wandb
from wandb.keras import WandbCallback

wandb.login()
run = wandb.init(project='CS6910_Assignment_I')
# Load dataset
(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

label_index, indices_list = np.unique(Y_train, return_index=True)
example_image = []
label_image = []
for i in range(len(label_index)):
  example_image.append(X_train[indices_list[i]])
  label_image.append(class_names[i])
print(label_image)

wandb.log({"fashion_mnist_examples": [wandb.Image(image, caption=label) for image,label in zip(example_image,label_image)]})



# Confusion matrix code

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
Y_test = np.load('')
cm =confusion_matrix(test_label, Y_test)  # test_label is model output, Y_test is ground truth
index = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
columns = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
cm_df = pd.DataFrame(cm,columns,index)
plt.figure(figsize=(12,8))
image = sns.heatmap(cm_df, annot=True)
wandb.log({"Confusion Matrix": [wandb.Image(image,caption='Confusion matrix for the best model')]})