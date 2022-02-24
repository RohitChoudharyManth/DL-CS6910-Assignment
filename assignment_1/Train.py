from assignment_1.ActivationLayerScalar import ActivationLayerScalar
from assignment_1.ActivationLayerFactory import tanh, tanh_d, sigmoid, sigmoid_d, softmax, softmax_d, relu, relu_d
from assignment_1.ActivationLayerVector import ActivationLayerVector
from assignment_1.Dense import Dense
from assignment_1.LossFactory import mse, mse_prime, cross_entropy, cross_entropy_d
from assignment_1.Network import Network
import numpy as np
from tensorflow.keras.datasets import fashion_mnist as mnist
from tensorflow.keras.utils import to_categorical

from assignment_1.Optimizer import Optimizer

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test_cat = to_categorical(y_test)



# Network
nw = Network()
nw.use(cross_entropy, cross_entropy_d, optimizer='SGD', learning_rate=0.001)
nw.add(Dense(28*28, 256))
nw.add(ActivationLayerScalar(sigmoid, sigmoid_d))
nw.add(Dense(256, 64))
nw.add(ActivationLayerScalar(sigmoid, sigmoid_d))
nw.add(Dense(64, 10))
nw.add(ActivationLayerVector(softmax, softmax_d))


nw.fit(x_train[:1000], y_train[:1000], x_test, y_test_cat, epochs=1, batch_size = 1000)


out = nw.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])