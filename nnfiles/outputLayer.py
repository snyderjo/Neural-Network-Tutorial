import numpy as np
from abc import ABC, abstractmethod
from activation_functions import logit
import nnlayer as nnl


class baseOuputLayerClassifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self,a_prev):
        pass

    @abstractmethod
    def loss(self,y):
        pass

    @abstractmethod
    def backprop(self,y):
        pass

    @abstractmethod
    def predict(self,a_prev):
        pass

    def update(self):
        pass


class classMultOutLayer(baseOuputLayerClassifier):
    ##needs a non-relu layer immediately before to enable negative activations
    def __init__(self,n_prev,n_class):
        self.nClass = nClass
        self.y_hat = None
        #add an alpha parameter?
        self.linLayer = nnl.baseHiddenLayer(n_self = n_class,n_prev = n_prev,act_funtion = linear)
        self.a_prev = None

    def forward(self,a_prev):
        self.a_prev = self.linLayer.forward(a_prev)
        self.y_hat = logit.f(a_prev)

    def loss(self,y):
        losses = - np.multiply(y, np.log(y_hat)) - np.multiply(1 - y, np.log(1 - y_hat))

        return np.mean(losses)

    def backprop(self,y):
        ##update this to reflect the added linear layer
        loss_grad = self.y_hat - y
        return self.linLayer.backprop(loss_grad)

    def predict(self):
        return y_hat

    def update(self):
        self.linLayer.update()


class classMutExcLayer(baseOuputLayerClassifier):
    #fiddle with n_prev or nClass?
    def __init__(self):
        self.y_hat = None
        self.a_prev = None

    @classmethod
    def softmax(cls,act_matrix):
        max_a = np.amax(act_matrix,axis = 0)
        exp_a = np.exp(act_matrix - max_a)
        sum_exp_a = np.sum(exp_a,0)

        return np.divide(exp_a,sum_exp_a)

    @classmethod
    def softmax_delta(cls,act_matrix):
        exp_a = np.exp(act_matrix)
        sum_exp_a = np.sum(exp_a,0)

        numerator = np.multiply(act_matrix, np.multiply(exp_a, (sum_exp_a - exp_a)))
        denominator = np.square(sum_exp_a)


    def forward(self,a_prev):
        self.a_prev = a_prev
        self.y_hat = cls.softmax(a_prev)

    def loss(self,y):
        """
        Assumes that y is a one-hot matrix of dimensions nClass x m
        """
        losses = -np.multiply(y, np.log(y_hat))

        return np.mean(losses)

    def backprop(self,y):
        dY_hat = np.divide(y,y_hat) #mostly zeros
        dA_prev = cls.softmax_delta(self.a_prev)

        return np.multiply(dY_hat, dA_prev)



    def predict(self,y):
        return y_hat




