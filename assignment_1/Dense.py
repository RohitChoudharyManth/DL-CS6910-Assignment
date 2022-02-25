import numpy as np
import copy
from assignment_1.BaseLayer import BaseLayer
from assignment_1.LayerHistory import LayerHistory
from assignment_1.Optimizer import Optimizer
from assignment_1.WeightInitializer import WeightInitializer

'''
Initialize this class with Dense(input_size, output_size)
input_size :- refers to the number of neurons in the previous layer.
output_size :- refers to the number of neurons in this Dense layer
'''

class Dense(BaseLayer):
    def __init__(self, input_size, output_size, initializer_type='random'):
        super(Dense, self).__init__()
        self.W = WeightInitializer().get_initial_weights(input_size, output_size, initializer_type=initializer_type)
        self.B = WeightInitializer().get_initial_bias(input_size, output_size, initializer_type=initializer_type)
        self.W_history = LayerHistory()
        self.B_history = LayerHistory()
        self.weights_error_list = []
        self.bias_error_list = []


    def forward(self, input):
        self.input = copy.deepcopy(input)
        self.output = np.matmul(self.input, self.W) + self.B
        return copy.deepcopy(self.output)


    def backward(self, output_error, w_optimizer, b_optimizer):
        inp_error = np.dot(output_error, self.W.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error
        self.weights_error_list.append(weights_error)
        self.bias_error_list.append(bias_error)
        return inp_error

    def step(self, w_optimizer, b_optimizer):
        average_batch_weight_error = np.mean(np.asarray(self.weights_error_list, dtype=np.float32), axis=0)
        average_batch_bias_error = np.mean(np.asarray(self.bias_error_list, dtype=np.float32), axis=0)
        self.W, self.W_history = w_optimizer.optimizer(copy.deepcopy(self.W), average_batch_weight_error, self.W_history)
        self.B, self.B_history = b_optimizer.optimizer(copy.deepcopy(self.B), average_batch_bias_error, self.B_history)
        self.weights_error_list.clear()
        self.bias_error_list.clear()
