import torch
import math
from torch import empty
import time
import datetime
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

#--------------------------------------------------#   
### General parent class ###
class Module(object) :
    """
    Class Module
    """
    def forward( self, *input):
        raise NotImplementedError
    def backward(self , *gradwrtoutput):
        raise NotImplementedError
    def getParams(self):
        return []

#--------------------------------------------------#   
### Sequential container ###

class Model(Module):
    """
    Class Model keeping the different modules and their parameters in memory.
    """
    def __init__(self):
        self.parameters= []
        self.modules = []
        
    
    def add(self,layer):
        self.modules.append(layer)
    
    def forward(self, x):
        out = x
        for module in self.modules:
            out = module.forward(out)
        return out
    
    def backward(self, output_grad):
        backwards_modules = self.modules[::-1]
        out = output_grad

        for module in backwards_modules:
            out = module.backward(out)
    
    def param(self):
        parameters = []

        for module in self.modules:
            parameters.append(module.getParams())

        return parameters
    
#--------------------------------------------------#   
### Operators  ###

class Linear(Module) :
    """
    Layer connector: Linear.
    """
    def __init__(self, in_ , out_):
        self.type = 'linear'
        
        self.weights = torch.Tensor(out_, in_).normal_(mean = 0 , std = 1)
        self.bias = torch.Tensor(out_).normal_(mean = 0, std = 1)
        
        self.grad_weights = torch.Tensor(self.weights.size())
        self.grad_bias = torch.Tensor(self.bias.size())

    def forward(self,x): 
        self.x = x
        return self.weights.mv(self.x) + self.bias

    def backward(self,d_y):
        self.grad_weights+=d_y.view(-1,1).mm(self.x.view(1,-1))
        self.grad_bias+=d_y
        return self.weights.t().mv(d_y)

    def update(self,lr):
        self.weights.add_(-lr * self.grad_weights)
        self.bias.add_(-lr * self.grad_bias)
        
    def zero_grad(self):
        self.grad_weights.zero_()
        self.grad_bias.zero_()
        

    def getParams(self):
        return [(self.weights, self.grad_weights),(self.bias, self.grad_bias)]

    
#--------------------------------------------------#   
### Activation functions ###

class Sigmoid(Module):
    """
    Activation module: Sigmoid.
    """
    def __init__(self):
        self.type='activation'
    
    def sigmoid(self, x):    
        return 1.0/(1.0 + torch.exp(-x)) 
    
    def sigmoid_derivative(self,x):
        return x * (1 - x) 
        
    def forward(self, x):
        self.x = x  
        return self.sigmoid(x)
    
    def backward(self, d_y):
        return self.sigmoid_derivative(self.x).mul(d_y)
       
    def getParams(self):
        return [(None, None)]


class ReLU(Module):
    """
    Activation module: ReLU.
    """
    def __init__(self):
        super(ReLU).__init__()
        self.type='activation'
    
    def forward(self, x):
        self.x = x
        relu = torch.clamp(self.x, min=0)
        return relu
    
    def backward(self, d_y):
        input_ = self.x
        relu_ = input_.sign().clamp(min = 0)
        grad = d_y * relu_
        return grad 
    
    def getParams(self):
        return [(None,None)]

class tanh(Module):
    """
    Activation module: tanh.
    """
    def __init__(self):
        super().__init__()
        self.type = 'activation'

    def forward(self,x):
        self.x = x
        output=torch.tanh(x)
        return output
    
    def tanh_(x):
        output = []
        for x in input:
            tanh = (2/ (1 + math.exp(-2*x))) -1 
            output.append(tanh)

        return torch.FloatTensor(tanh)

    def backward(self,d_y):
        return 4*((self.x.exp() + self.x.mul(-1).exp()).pow(-2)) * d_y

    def getParams(self):
        return [(None, None)]                      


#--------------------------------------------------#   
### Loss functions ###

class LossMSE(Module):
    """
    Loss Function module: MSE.
    """
    def _init_(self):
        super(LossMSE)._init_()
        self.name = 'LossMSE'

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return (pred - target.float()).pow(2).mean()

    def backward(self):
        return 2*(self.pred - self.target.float())


class LossMAE(Module):
    """
    Loss Function module: MAE.
    """
    def __init__(self):
        super(LossMAE).__init__()
        self.name= 'MAE'

    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return (pred - target).abs().mean()

    def backward(self):
        return (self.pred - self.target).sign()


#--------------------------------------------------#   
### Optimizer ###

class SGD():
    """
    Optimizer module: SGD
    """
    def __init__(self, model, learning_rate):
        self.model= model
        self.lr = learning_rate
    
    def step(self):
        for module in self.model.modules:
            if module.type=='linear':
                module.update(self.lr)
        
    
    def zero_grad(self):
        for module in self.model.modules:
            if module.type=='linear':
                module.zero_grad()    
