'''
Code all the activation layers and their derivative functions
'''
import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    return 1-np.tanh(x)**2
