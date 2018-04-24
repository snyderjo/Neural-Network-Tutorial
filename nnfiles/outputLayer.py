import numpy as np
from abc import ACB, abstractmethod
from activation_functions.logit import f as logit

class baseOuputLayer(abc)
    @abstractmethod
    def __init__(self):
        self.y_hat = None

    @abstractmethod
    def forward(self,a_prev):
        self.y_hat = None

    @abstractmethod
    def loss(self,y):
        pass

    @abstractmethod
    def backprop(self,y):
        pass

    @abstractmethod
    def predict(self,y):
        pass