import numpy as np

from assignment_1.Optimizer import Optimizer
from sklearn.metrics import accuracy_score


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.w_optimizer = None
        self.b_optimizer = None
        #self.w_history = []
        #self.b_history = []

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime, optimizer='SGD', learning_rate=1e-2, gamma=0.9,
            epsilon=1e-8, beta1=0.9, beta2=0.999):
        self.loss = loss
        self.loss_prime = loss_prime
        self.w_optimizer = Optimizer(optimizer=optimizer, learning_rate=learning_rate, gamma=gamma,
                                     epsilon=epsilon, beta1=beta1, beta2=beta2)
        self.b_optimizer = Optimizer(optimizer=optimizer, learning_rate=learning_rate, gamma=gamma,
                                     epsilon=epsilon, beta1=beta1, beta2=beta2)

        # predict output for given input

    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward(output)
            np.argmax(output)
            result.append(np.argmax(output))
        op_arr = np.asarray(result)
        return op_arr

    # train the network
    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size=5):
        # sample dimension first
        # samples = len(x_train)

        #iterating over number of batches
        No_of_batches = int(len(x_train) / batch_size)

        # training loop
        for i in range(epochs):
            err = 0

            for n in range(No_of_batches):
                x_train_batch = x_train[n * batch_size: (n + 1) * batch_size]
                y_train_batch = y_train[n * batch_size: (n + 1) * batch_size]
                samples = len(x_train_batch)


                for j in range(samples):
                    # forward propagation

                    # training sample taken as output of the input layer
                    output = x_train_batch[j]
                    for layer in self.layers:
                        output = layer.forward(output)

                    # compute loss (for display purpose only)
                    err += self.loss(y_train_batch[j], output)

                    #the idea is to set a flag = 1 when one batch ends
                    num_of_points_seen = 0

                    if num_of_points_seen % batch_size ==0 :
                        flag = 1
                    else:
                        flag = 0

                    #the idea is to use flag only for dense layers and not activation scalar/vector
                    layer_index: int = 0
                    # backward propagation
                    error = self.loss_prime(y_train_batch[j], output)
                    for layer in reversed(self.layers):
                        if layer_index % 2 ==0 :
                            error = layer.backward(error, self.w_optimizer, self.b_optimizer)
                        else:
                            error = layer.backward(error, self.w_optimizer, self.b_optimizer)

                        layer_index = layer_index + 1
                    # pass

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))
            y_pred_train = self.predict(x_train)
            train_acc = accuracy_score(np.squeeze(np.argmax(y_train, axis=1)), np.squeeze(y_pred_train))
            print('epoch %d/%d   train_accuracy=%f' % (i + 1, epochs, train_acc))

            y_pred_test = self.predict(x_test)
            test_acc = accuracy_score(np.squeeze(np.argmax(y_test, axis=1)), np.squeeze(y_pred_test))
            print('epoch %d/%d   train_accuracy=%f' % (i + 1, epochs, test_acc))
