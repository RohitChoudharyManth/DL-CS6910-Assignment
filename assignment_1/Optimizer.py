'''
Optimizer class, implement all the optimizers as methods of this class and add it in the select list
'''


class Optimizer:
    def __init__(self,  optimizer='SGD', learning_rate = 1e-2, gamma = 0.9, epsilon = 1e-8, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.update = 0
        self.m = 0
        self.v = 0

        if optimizer == 'SGD':
            self.optimizer = self.SGD_optim
        elif optimizer == 'Momentum':
            self.optimizer = self.momentum_optim
        elif optimizer == 'nesterov':
            self.optimizer = self.nesterov_optim
        elif optimizer == 'rmsprop':
            self.optimizer = self.rmsprop_optim
        elif optimizer == 'adam':
            self.optimizer == self.adam_optim
        elif optimizer == 'nadam':
            self.optimizer = self.nadam_optim
        else:
            print('Give correct optimizer name')
            raise Exception

    # def SGD_optim(self, W, B, W_error, B_error):
    #    W -= self.lr * W_error
    #    B -= self.lr * B_error
    #    return W, B

    def SGD_optim(self, param, param_grad):
        param -= self.lr * param_grad
        return param

    def momentum_optim(self, param, param_grad):
        self.update = self.gamma * self.update + self.lr * param_grad
        param -= self.update
        return param

    def nesterov_optim(self, param, param_grad):
        param_lookahead = param - self.gamma * self.update
        self.update = self.gamma * self.update + self.lr *  ##### take gradient of param_lookahead
        param -= self.update
        return param
