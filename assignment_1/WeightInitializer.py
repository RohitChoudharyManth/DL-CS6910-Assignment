import numpy as np
'''
This is a function repository, all the initializers will be implemented here as functions
'''

def get_weights_random(input_size, output_size):
    return np.random.randn(input_size, output_size)

def get_bias_random(output_size):
    return np.random.randn(1, output_size)
