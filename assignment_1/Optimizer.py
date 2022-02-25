from assignment_1.LayerHistory import LayerHistory
import numpy as np

class Optimizer:
    def __init__(self,  optimizer='SGD', learning_rate = 1e-2, gamma = 0.9, epsilon = 1e-8, beta1=0.9, beta2=0.999):
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        # beta and beta 1 are taken as same constants
        self.beta1 = beta1
        self.beta2 = beta2
        if optimizer == 'SGD':
            self.optimizer = self.SGD_optim
        elif optimizer == 'Momentum':
            self.optimizer = self.momentum_optim
        elif optimizer == 'nesterov':
            self.optimizer = self.nesterov_optim
        elif optimizer == 'rmsprop':
            self.optimizer = self.rmsprop_optim
        elif optimizer == 'adam':
            self.optimizer = self.adam_optim
        elif optimizer == 'nadam':
            self.optimizer = self.nadam_optim
        else:
            print('Give correct optimizer name')
            raise Exception


    def SGD_optim(self, param, param_grad, param_history):
        param -= self.lr * param_grad
        return param, param_history

    def momentum_optim(self, param, param_grad, param_history: LayerHistory):

        param_history.update = self.gamma * param_history.update + self.lr * param_grad
        param -= param_history.update
        return param, param_history

    # # def nesterov_optim(self, param, param_grad):
    #     param_lookahead = param - self.gamma * self.update
    #     self.update = self.gamma * self.update + self.lr * param_grad  ##### take gradient of param_lookahead
    #     param -= self.update
    #     return param

    def rmsprop_optim(self, param, param_grad, param_history):
        param_history.v = self.beta1 * param_history.v + (1 - self.beta1) * (param_grad**2)
        param -= ((self.lr)/((param_history.v + self.epsilon)**(1/2))) * param_grad
        return param, param_history

    def adam_optim(self, param, param_grad, param_history):
        param_history.m = self.beta1 * param_history.m +(1 - self.beta1) * param_grad
        param_history.v = self.beta2 * param_history.v +(1 - self.beta2) * (param_grad ** 2)
        m_hat = param_history.m / (1 - (self.beta1 ** param_history.t))
        v_hat = param_history.v / (1 - (self.beta2 ** param_history.t))
        param -= np.divide(self.lr, np.sqrt(v_hat) + self.epsilon) * m_hat
        param_history.t += 1
        return param, param_history

## referenced from https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9

    def nadam_optim(self, param, param_grad, param_history):
        param_history.m = self.beta1 * param_history.m + (1 - self.beta1) * param_grad
        param_history.v = self.beta2 * param_history.v + (1-self.beta2) * (param_grad**2)
        m_hat = param_history.m / (1 - (self.beta1 ** param_history.t))
        v_hat = param_history.v / (1 - (self.beta2 ** param_history.t))
        param -= ((self.lr)/((v_hat + self.epsilon)**(1/2))) * (self.beta1 * m_hat + ((1-self.beta1)/(1-self.beta1 ** param_history.t)) * param_grad)
        param_history.iter += 1
        return param, param_history
