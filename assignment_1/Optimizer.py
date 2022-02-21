'''
Optimizer class, implement all the optimizers as methods of this class and add it in the select list
'''


class Optimizer:
    def __init__(self, learning_rate, optimizer='SGD'):
        self.lr = learning_rate

        if optimizer == 'SGD':
            self.optimizer = self.SGD_optim
        else:
            print('Give correct optimizer name')
            raise Exception

    def SGD_optim(self, W, B, W_error, B_error):
        W -= self.lr*W_error
        B -= self.lr * B_error
        return W, B