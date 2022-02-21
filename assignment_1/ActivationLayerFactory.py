'''
Code all the activation layers and their derivative functions
'''
import numpy as np


def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    return 1-np.tanh(x)**2


def sigmoid(x):
    return np.divide(1, (1+np.exp(-x)))


def sigmoid_d(x):
    return np.multiply(sigmoid(x), 1 - sigmoid(x))


def relu(x):
    return np.clip(x, 0, None)


def relu_d(x):
    return np.where(x > 0, 1, 0)


def softmax(x, eps=1e-6):
    return np.true_divide(np.exp(x - np.max(x)), np.sum(np.exp(x - np.max(x))))


def softmax_d(x, y):
    return np.multiply(y, np.identity(y.shape[0]) - y)