import numpy as np

'''
Designing this MLP with two hidden layers and a binary output,
trained on cross entropy loss
inp_dim = R^3
output_dim = R^1

hidden_layer_list = [2, 2]

W_1 = [2, 4]
W_2 = [2, 3]
W_3 = [1, 3]

'''


class MLP:
    def __init__(self):
        self.W_1 = np.ones((2, 4))
        self.W_2 = np.ones((2, 3))
        self.W_3 = np.ones((1, 3))

    def activation_func(self, x):
        return np.divide(np.exp(x), np.exp(x) + 1)

    def forward(self, x):
        h_1 = self.activation_func(np.matmul(self.W_1, x))
        h_2 = self.activation_func(np.matmul(self.W_2, np.vstack((np.ones((1, 1)), h_1))))
        h_3 = self.activation_func(np.matmul(self.W_3, np.vstack((np.ones((1, 1)), h_2))))
        return h_3

x = np.random.randn(3, 1)
print(x)
x = np.vstack((np.ones((1, 1)), x))
mlp = MLP()
print(mlp.forward(x))