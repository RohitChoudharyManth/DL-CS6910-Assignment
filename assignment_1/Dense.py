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
    def __init__(self, input_size, output_size):
        super(Dense, self).__init__()
        self.W = WeightInitializer().get_initial_weights(input_size, output_size, initializer_type='random')
        self.B = WeightInitializer().get_initial_bias(input_size, output_size, initializer_type='random')
        self.W_history = LayerHistory()
        self.B_history = LayerHistory()


    def forward(self, input):
        self.input = copy.deepcopy(input)
        self.output = np.matmul(self.input, self.W) + self.B
        return copy.deepcopy(self.output)


    def backward(self, output_error, w_optimizer, b_optimizer):
        inp_error = np.dot(output_error, self.W.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error
        self.W, self.W_history = w_optimizer.optimizer(copy.deepcopy(self.W), weights_error, self.W_history)
        self.B, self.B_history = b_optimizer.optimizer(copy.deepcopy(self.B), bias_error, self.B_history)
        return inp_error
