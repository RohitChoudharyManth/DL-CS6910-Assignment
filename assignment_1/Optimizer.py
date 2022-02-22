class Optimizer:
    def __init__(self,  optimizer='SGD', learning_rate = 1e-2, gamma = 0.9, epsilon = 1e-8, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # beta and beta 1 are taken as same constants
        self.beta1 = beta1
        self.beta2 = beta2
        self.update = 0
        self.m = 0
        self.v = 0
        self.t = 1

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

    #def SGD_optim(self, W, B, W_error, B_error):
    #    W -= self.lr * W_error
    #    B -= self.lr * B_error
    #    return W, B

    def SGD_optim(self, param, param_grad):
        param -= self.lr * param_grad
        return param

    def momentum_optim(self, param, param_grad):
        self.update=self.gamma * self.update + self.lr * param_grad
        param -= self.update
        return param

    # def nesterov_optim(self, param, param_grad):
    #     param_lookahead = param - self.gamma * self.update
    #     self.update = self.gamma * self.update + self.lr * param_grad  ##### take gradient of param_lookahead
    #     param -= self.update
    #     return param

    def rmsprop_optim(self, param, param_grad):
        self.v = self.beta1 * self.v + (1 - self.beta1) * (param_grad**2)
        param -= ((self.lr)/((self.v + self.epsilon)**(1/2))) * param_grad
        return param

    def adam_optim(self, param, param_grad):
        self.m = self.beta1 * self.m +(1 - self.beta1) * param_grad
        self.v = self.beta2 * self.v +(1 - self.beta2) * (param_grad ** 2)
        m_hat = self.m / (1 - (self.beta1 ** self.t))
        v_hat = self.m / (1 - (self.beta2 ** self.t))
        param -= ((self.lr)/((v_hat + self.epsilon)**(1/2))) * m_hat
        self.t += 1
        return param

## referenced from https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9

    def nadam_optim(self, param, param_grad):
        self.m = self.beta1 * self.m + (1 - self.beta1) * param_grad
        self.v = self.beta2 * self.v + (1-self.beta2) * (param_grad**2)
        m_hat = self.m / (1- (self.beta1 ** self.t))
        v_hat = self.v / (1- (self.beta2 ** self.t))
        param -= ((self.lr)/((v_hat + self.epsilon)**(1/2))) * (self.beta1 * m_hat + ((1-self.beta1)/(1-self.beta1 ** self.t)) * param_grad)
        self.iter += 1
        return param
