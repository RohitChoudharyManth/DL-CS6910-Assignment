import numpy as np
'''
This is a function repository, all the initializers will be implemented here as functions
'''

class WeightInitializer:
    def __init__(self):
        pass


    def get_initial_weights(self, input_size, output_size, initializer_type='random'):
        if initializer_type == 'random':
            return self.get_weights_random(input_size, output_size)
        elif initializer_type == 'xavier':
            return self.get_weights_xavier(input_size, output_size)
        else:
            raise NotImplementedError

    def get_initial_bias(self, input_size, output_size, initializer_type='random'):
        if initializer_type == 'random':
            return self.get_bias_random(input_size, output_size)
        elif initializer_type == 'xavier':
            return self.get_bias_xavier(input_size, output_size)
        else:
            raise NotImplementedError


    def get_weights_random(self, input_size, output_size):
        return np.random.randn(input_size, output_size)

    def get_bias_random(self, input_size, output_size):
        return np.random.rand(1, output_size)

    def get_weights_xavier(self, input_size, output_size):
        #normalized xavier weight intialization
        # calculate the range for the weights
        lower, upper = -(np.sqrt(6.0) / np.sqrt(input_size + output_size)), (np.sqrt(6.0) / np.sqrt(input_size + output_size))
        return np.random.uniform(low=lower, high=upper, size=(input_size, output_size))

    def get_bias_xavier(self, input_size, output_size):
        lower, upper = -(np.sqrt(6.0) / np.sqrt(input_size + output_size)), (np.sqrt(6.0) / np.sqrt(input_size + output_size))
        return np.random.uniform(low=lower, high=upper, size=(1, output_size))
