import numpy as np
'''
This is a function repository, all the initializers will be implemented here as functions
'''

class Random_weight_bias():
    def __init__(self):
        pass

    def get_weights_random(self,input_size,output_size):
        return np.random.randn(input_size, output_size)

    def get_bias_random(self,output_size):
        return np.random.rand(1, output_size)

class Xavier_weight_bias():
    def __init__(self):
        pass

    def get_weights_xavier(self,input_size,output_size):
        #normalized xavier weight intialization
        # calculate the range for the weights
        lower, upper = -(sqrt(6.0) / sqrt(input_size + output_size)), (sqrt(6.0) / sqrt(input_size + output_size))
        return np.random.uniform(low=lower, high=upper, size=(input_size, output_size))

    def get_bias_xavier(self,input_size,output_size):
        lower, upper = -(sqrt(6.0) / sqrt(input_size + output_size)), (sqrt(6.0) / sqrt(input_size + output_size))
        return np.random.uniform(low=lower, high=upper, size=(1, output_size))


    #write for xavier


#def get_weights_random(input_size, output_size):
#    return np.random.randn(input_size, output_size)


#def get_bias_random(output_size):
#    return np.random.randn(1, output_size)
