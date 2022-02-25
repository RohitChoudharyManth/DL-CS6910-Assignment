# DL-CS6910-Assignment

# CS6910: Assignment 1
# Authors: EE20S086: Aditya Kanade, EE20S002: Rohit Choudhary
Scratch code for Feed Forward Neural Network

## Description of Files
1. To implement results along with wandb compilation wandb_run_assignment_1.ipynb"
2. To replicate results with different configurations refer "assignment_1/Train.py"
3. "Question_1_and_confusion_matrix.py" contains code to generate the image deck for Question 1 and the confusion matrix for the best performing model.

### wandb_run_assignment_1.ipynb(with wandb)
Here sweep configuration (.yaml file) is setted up, you all just need to provide wandb login key and if you want change sweep configuration edit this section:
```
The default sweep config is set to
  default_configs = {"epochs": 3 ,
            "learning_rate": 0.001,
            "No_hidden_layer": 3,
            "Neurons": 32,
            "weight_decay": 0,
            "mini_batch_size": 64,
            "weight_initialization": 'xavier',
            "activation_type": "relu",
            "loss_optimizer_type": "adam",
            }
While the sweep config is setup in the dictionary *sweep_config*

sweep_config = {
    'method': 'random', #grid, random
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values': [5,8]
        },
        'No_hidden_layer': {
            'values': [3,4,5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4, 1e-2]
        },'Neurons':{
            'values': [32,64,128]
        },
        'weight_decay': {
            'values': [0,0.0005,0.5]
        },'mini_batch_size':{
            'values': [16,32,64]
        },'loss_optimizer_type':{
            'values': ['rmsprop','adam','nadam', 'SGD', 'Momentum', 'NAG']
        },'weight_initialization': {
            'values': ['random','xavier']
        },'activation_type':{
            'values': ['sigmoid','tanh', 'relu']
        }
    }
}
```

### Train.py

To replicate results for a particular configuration, changes to the code can be made in the Train.py file. We have made efforts to write code in Keras way
so adding blocks of layers or activation fuction is very simple. Below is a small snippet of the training code.
```
nw = Network()
nw.use(mse, mse_prime, optimizer='nadam', learning_rate=1e-3)
nw.add(Dense(28*28, 100, initializer_type='xavier'))
nw.add(ActivationLayerScalar(activation='relu'))
nw.add(Dense(100, 50, initializer_type='xavier'))
nw.add(ActivationLayerScalar(activation='relu'))
nw.add(Dense(50, 10, initializer_type='xavier'))
nw.add(ActivationLayerVector(softmax, softmax_d))
nw.fit(x_train, y_train, x_test, y_test_cat, epochs=5, batch_size=128)
```

One can add as many layers as they want, Please note structure of Dense layer is as follows Dense(input_neuron_size, output_neuron_size)

Note:- One cad add new layers as well, all layers inherit the BaseLayer class. Each layer class should implement forward, backward and step methods.
As an example below is a snippet for the Dense class

```
class Dense(BaseLayer):
    def __init__(self, input_size, output_size, initializer_type='random'):
        super(Dense, self).__init__()
        self.W = WeightInitializer().get_initial_weights(input_size, output_size, initializer_type=initializer_type)
        self.B = WeightInitializer().get_initial_bias(input_size, output_size, initializer_type=initializer_type)
        self.W_history = LayerHistory()
        self.B_history = LayerHistory()
        self.weights_error_list = []
        self.bias_error_list = []


    def forward(self, input):
        self.input = copy.deepcopy(input)
        self.output = np.matmul(self.input, self.W) + self.B
        return copy.deepcopy(self.output)


    def backward(self, output_error, w_optimizer, b_optimizer):
        inp_error = np.dot(output_error, self.W.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error
        self.weights_error_list.append(weights_error)
        self.bias_error_list.append(bias_error)
        return inp_error

    def step(self, w_optimizer, b_optimizer):
        average_batch_weight_error = np.mean(np.asarray(self.weights_error_list, dtype=np.float32), axis=0)
        average_batch_bias_error = np.mean(np.asarray(self.bias_error_list, dtype=np.float32), axis=0)
        self.W, self.W_history = w_optimizer.optimizer(copy.deepcopy(self.W), average_batch_weight_error, self.W_history)
        self.B, self.B_history = b_optimizer.optimizer(copy.deepcopy(self.B), average_batch_bias_error, self.B_history)
        self.weights_error_list.clear()
        self.bias_error_list.clear()
```



Here input_neuron_size refers to the neurons in the previous layer, while output neuron size refers to the neurons in the current layer.
Activation function can be changed in the ActivationLayerScalar, by giving either of 'sigmoid', 'tanh', 'relu' options.

Adding new activations is easy, please add two methods in the ActivationFactory.py file implementing forward pass and gradient for the activation function. 
Below is a snippet for ReLU implementation

```
#called in forward pass, y=f(x), where x is input to the activation layer and y is the computed output.
def relu(x, eps=1e-8):
    return np.clip(x, eps, None)

#called in backward pass, calculates del(y)/del(x)
def relu_d(x, eps=1e-8):
    return np.where(x > 0, 1, eps)
```

Similarly, adding new loss functions is easy, please add two methods in the LossFactory.py file implementing loss w.r.t predicted vector and del(Loss)/del(y_pred)
Below is a snippet for Cross Entropy implementation

```
#Loss(y_true, y_pred)
def cross_entropy(y_true, y_pred, eps=1e-8):
    return np.where(y_true == 1, -np.log2(eps + y_pred)/y_true.size, 0).sum()

#del(Loss)/del(y_pred)
def cross_entropy_d(y_true, y_pred, eps=1e-8):
    return np.where(y_true == 1, -1/(eps+y_pred), 0)
```

### Question_1_and_confusion_matrix.py

In this file code for question 1 and confusion matrix is there.
Note: For confusion matrix Y_test and test_labels should be provided.
The Y_test generated for the best model is saved as best_y_test.npy in the assignment_1 folder
<!-- 
### [Check out our project workspace](https://wandb.ai/vrunda/CS6910_Assignment1?workspace=user-vrunda)
### [Check out project detailed report](https://wandb.ai/vrunda/CS6910_Assignment1/reports/CS6910-Assignment-1--Vmlldzo1MzI4NjE) -->
