import copy

from numpy import ndarray

from assignment_1.BaseLayer import BaseLayer
import numpy as np

class ActivationLayer(BaseLayer):
    def __init__(self, activation, activation_d):
        super(ActivationLayer, self).__init__()
        self.activation = activation
        self.activation_d = activation_d

    def forward(self, input : ndarray):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_error, optimizer):
        return np.multiply(output_error, self.activation_d(self.input))