from sklearn.metrics import accuracy_score

from assignment_1.ActivationLayerScalar import ActivationLayerScalar
from assignment_1.ActivationLayerFactory import tanh, tanh_d, sigmoid, sigmoid_d, softmax, softmax_d, relu, relu_d
from assignment_1.ActivationLayerVector import ActivationLayerVector
from assignment_1.Dense import Dense
from assignment_1.LossFactory import mse, mse_prime, cross_entropy, cross_entropy_d
from assignment_1.Network import Network
import numpy as np

import numpy as np
np.random.seed(10)
from matplotlib import pyplot
from keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

from tensorflow.keras.datasets import fashion_mnist as mnist
from tensorflow.keras.utils import to_categorical

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



from sklearn.metrics import accuracy_score

from assignment_1.ActivationLayerScalar import ActivationLayerScalar
from assignment_1.ActivationLayerFactory import tanh, tanh_d, sigmoid, sigmoid_d, softmax, softmax_d, relu, relu_d
from assignment_1.ActivationLayerVector import ActivationLayerVector
from assignment_1.Dense import Dense
from assignment_1.LossFactory import mse, mse_prime, cross_entropy, cross_entropy_d
from assignment_1.Network import Network
import numpy as np
from tensorflow.keras.datasets import fashion_mnist as mnist
from tensorflow.keras.utils import to_categorical




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
nw.use(mse, mse_prime, optimizer='nadam', learning_rate=1e-3)
nw.add(Dense(28*28, 100, initializer_type='xavier'))
nw.add(ActivationLayerScalar(activation='relu'))
nw.add(Dense(100, 50, initializer_type='xavier'))
nw.add(ActivationLayerScalar(activation='relu'))
nw.add(Dense(50, 10, initializer_type='xavier'))
nw.add(ActivationLayerVector(softmax, softmax_d))


nw.fit(x_train, y_train, x_test, y_test_cat, epochs=5, batch_size=128)


y_test_pred = nw.predict(x_test)
test_accuracy = accuracy_score(y_test, np.squeeze(y_test_pred))

print("\n")
print('Test Set Score ', test_accuracy)