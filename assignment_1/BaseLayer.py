
class BaseLayer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def step(self,  w_optimizer, b_optimizer):
        pass