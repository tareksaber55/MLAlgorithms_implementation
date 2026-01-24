import numpy as np
from numpy.linalg import norm

class NeuralNetwork():
    def __init__(self,hidden_layer_weights,output_layer_weights,act_types):
        self.hidden_layer = Layer(len(hidden_layer_weights),act_types[0],hidden_layer_weights)
        self.output_layer = Layer(len(output_layer_weights),act_types[1],output_layer_weights)
        self.output_layer_weights = output_layer_weights
        self.hidden_layer_weights = hidden_layer_weights

    def feed_forward(self,input):
        self.hidden_layer.feed_forward(input)
        outputs = np.array([neuron.out for neuron in self.hidden_layer.neurons])
        self.output_layer.feed_forward(outputs)

    def compute_delta(self,t):
        outputs = np.array([neuron.out for neuron in self.output_layer.neurons])
        gradients_out = np.array([neuron.compute_act_derv() for neuron in self.output_layer.neurons])
        out_net = (t - outputs) * gradients_out
        gradients_hidden = np.array([neuron.compute_act_derv() for neuron in self.hidden_layer.neurons])
        hidden_net = (self.output_layer_weights.T @ out_net) * gradients_hidden
        return out_net , hidden_net
    
    def update_weights(self,out_net,hidden_net,input):
        outputs_hidden = np.array([neuron.out for neuron in self.hidden_layer.neurons])
        lr = 0.1
        for i, neuron in enumerate(self.output_layer.neurons):
            neuron.weights -= lr * out_net[i] * outputs_hidden

        for i, neuron in enumerate(self.hidden_layer.neurons):
            neuron.weights -= lr * hidden_net[i] * input
        
    def train_step(self, x, t, tol=0.01, max_iters=1000):
        output_layer_prev = self.output_layer_weights + 100
        hidden_layer_prev = self.hidden_layer_weights + 100

        i = 0
        it = 0
        while (norm(self.output_layer_weights - output_layer_prev) > tol or
            norm(self.hidden_layer_weights - hidden_layer_prev) > tol) and it < max_iters:

            output_layer_prev = self.output_layer_weights.copy()
            hidden_layer_prev = self.hidden_layer_weights.copy()

            self.feed_forward(x[i])
            out_net, hidden_net = self.compute_delta(t[i])
            self.update_weights(out_net, hidden_net, x[i])

            # refresh stored weight matrices
            self.output_layer_weights = np.array([n.weights for n in self.output_layer.neurons])
            self.hidden_layer_weights = np.array([n.weights for n in self.hidden_layer.neurons])

            i = (i + 1) % len(x)
            it += 1

class Layer():
    def __init__(self,neuron_num,act_type,weights):
        self.neuron_num = neuron_num
        self.act_type = act_type
        self.weights = weights
        self.prep_layer()

    def prep_layer(self):
        self.neurons = []
        for i in range(self.neuron_num):
            neuron =  Neuron(self.weights[i],self.act_type)   
            self.neurons.append(neuron)
            
    def feed_forward(self,input):
        for i in range(self.neuron_num):
            self.neurons[i].calc_net_out(input)
            
class Neuron():
    def __init__(self,weights,act_type):
        self.act_type = act_type
        self.weights = weights.astype(float)
    def compute_activation(self,net):
        if self.act_type == 'sigmoid':
            return 1 / (1 + np.exp(-net))        
        elif self.act_type == 'tanh':
            return (1 - np.exp(-2*net)) / (1 + np.exp(-2*net))
        elif self.act_type == 'identity':
            return net
        else :
            return max(0,net)
    def compute_act_derv(self):
        if self.act_type == 'sigmoid':
            return self.out * (1 - self.out)
        elif self.act_type == 'tanh':
            return 1 - self.out**2
        elif self.act_type == 'identity':
            return 1
        else:
            return np.where(self.out>0,1,0)
    def calc_net_out(self,input):
        self.input = input
        self.net = np.dot(self.input, self.weights)
        self.out = self.compute_activation(self.net)
        