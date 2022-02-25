import copy

from numpy import ndarray

from assignment_1.ActivationLayerFactory import sigmoid, sigmoid_d, tanh, tanh_d
from assignment_1.BaseLayer import BaseLayer
import numpy as np

def get_activation_func(activation):
    if activation == 'sigmoid':
        return sigmoid, sigmoid_d
    elif activation == 'tanh':
        return tanh, tanh_d
    else:
        raise NotImplementedError


class ActivationLayerScalar(BaseLayer):
    def __init__(self, activation='sigmoid'):
        super(ActivationLayerScalar, self).__init__()
        self.activation, self.activation_d = get_activation_func(activation=activation)

    def forward(self, input : ndarray):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_error, w_optimizer, b_optimizer):
        return np.multiply(output_error, self.activation_d(self.input))