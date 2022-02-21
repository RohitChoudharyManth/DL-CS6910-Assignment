import numpy as np


def get_weights_random(input_size, output_size):
    return np.random.randn(input_size, output_size)

def get_bias_random(output_size):
    return np.random.randn(1, output_size)
