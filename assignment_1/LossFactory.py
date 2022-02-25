
import numpy as np

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


def cross_entropy(y_true, y_pred, eps=1e-8):
    return np.where(y_true == 1, -np.log2(eps + y_pred)/y_true.size, 0).sum()

def cross_entropy_d(y_true, y_pred, eps=1e-8):
    return np.where(y_true == 1, -1/(eps+y_pred), 0)

