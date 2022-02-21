from assignment_1.ActivationLayer import ActivationLayer
from assignment_1.ActivationLayerFactory import tanh, tanh_d, relu, relu_d
from assignment_1.Dense import Dense
from assignment_1.LossFactory import mse, mse_prime
from assignment_1.Network import Network
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from assignment_1.Optimizer import Optimizer

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)



# Network
nw = Network()
optimizer = Optimizer(learning_rate=0.1, optimizer='SGD')
nw.use(mse, mse_prime, optimizer)
nw.add(Dense(28*28, 100))
nw.add(ActivationLayer(tanh, tanh_d))
nw.add(Dense(100, 50))
nw.add(ActivationLayer(tanh, tanh_d))
nw.add(Dense(50, 10))
nw.add(ActivationLayer(tanh, tanh_d))

nw.fit(x_train[:1000], y_train[:1000], epochs=100)


out = nw.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])