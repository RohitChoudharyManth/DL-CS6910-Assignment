import copy

from numpy import ndarray

from assignment_1.BaseLayer import BaseLayer
import numpy as np


class ActivationLayerVector(BaseLayer):
    def __init__(self, activation, activation_d):
        super(ActivationLayerVector, self).__init__()
        self.activation = activation
        self.activation_d = activation_d

    def forward(self, input : ndarray):
        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward(self, output_error, w_optimizer, b_optimizer):
        return np.multiply(output_error, self.activation_d(self.input, self.output))