import numpy as np
from abc import ABC, abstractmethod
from activation_functions import logit


class baseOuputLayerClassifier(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self,A_prev):
        pass

    @abstractmethod
    def loss(self,y):
        pass

    @abstractmethod
    def backprop(self,y):
        pass

    @abstractmethod
    def predict(self,A_prev):
        pass

    @abstractmethod
    def predictClass(self):
        pass

    def updateAlpha(self,alpha):
        pass

class classMultOutLayer(baseOuputLayerClassifier):
    ##needs a non-relu layer immediately before to enable negative activations
    def __init__(self,classDict):
        self.classDict = classDict
        self.y_hat = np.zeros([], dtype = np.float128)
        self.A_prev = np.zeros([], dtype = np.float128)

    def forward(self,A_prev):
        self.y_hat = logit.f(A_prev)

    def loss(self,y):
        losses = - np.add(
            np.multiply(y, np.log(self.y_hat))
            , np.multiply(
                np.subtract(1 , y)
                , np.log(np.subtract(1 , self.y_hat))
                )
            )
            # - y * log(y_hat) - (1 - y) * log(1 - y_hat)

        return np.mean(losses)

    def backprop(self,y):
        loss_grad = np.subtract(self.y_hat , y)
        return loss_grad

    def predict(self):
        return y_hat

    def predictClasses(self,threshold = .5):
        pass

    def predictClass(self):
        max_ind = np.argmax(self.y_hat,axis = 0)
        max_prob = np.amax(self.y_hat,axis = 0)
        max_prob_class = [self.classDict[x] for x in max_ind]

        return zip(max_ind,max_prob_class,max_prob)


class classMutExcLayer(baseOuputLayerClassifier):
    #fiddle with n_prev or nClass?
    def __init__(self,classDict):
        self.classVec = classDict
        self.y_hat = np.zeros([], dtype = np.float128)
        self.A_prev = np.zeros([], dtype = np.float128)

    @classmethod
    def softmax(cls,act_matrix):
        max_a = np.amax(act_matrix,axis = 0)
        exp_a = np.exp(act_matrix - max_a)
        sum_exp_a = np.sum(exp_a, axis = 0)

        return np.divide(exp_a,sum_exp_a)

    @classmethod
    def softmax_delta(cls,act_matrix,y):
        max_a = np.amax(act_matrix, axis = 0)
        exp_a = np.exp(np.subtract(act_matrix , max_a))
        sum_exp_a = np.sum(exp_a, axis = 0)

        #the gradient when the activation is in the softmax numerator
        numerator_numer = np.multiply(exp_a, (np.subtract(sum_exp_a , exp_a)))
        denominator_numer = np.square(sum_exp_a)

        #the gradient in the event activation is not in the softmax numerator
        #Still need to take what the sofmax numerator of the y_hat eventually became, i.e. that of y
        softmax_numerator = np.max(np.multiply(exp_a, y) ,axis = 0) #exp(a) is guaranteed to be positive, and the non-numerator values will be zero
        numerator_denom = -np.multiply(softmax_numerator,exp_a)
        denominator_denom = np.square(sum_exp_a)

        soft_delta  = np.add(
            np.multiply(y,np.divide(numerator_numer,denominator_numer))
            , np.multiply(
                np.subtract(1 , y)
                ,np.divide(numerator_denom,denominator_denom)
                )
            )# == y * (numerator_delta)  + (1 - y) * (denominator_delta)


        return soft_delta


    def forward(self,A_prev):
        self.A_prev = A_prev
        self.y_hat = self.softmax(A_prev)
        #alter near-zero values of Y_hat to slightly positive to prevent errors?


    def loss(self,y):
        """
        Assumes that y is a one-hot matrix of dimensions nClass x m
        """
        losses = -np.multiply(y, np.log(self.y_hat))

        return np.mean(losses)

    def backprop(self,y):
        dY_hat = -np.divide(y,self.y_hat) #mostly zeros

        dA_prev = self.softmax_delta(self.A_prev,y)

        return np.multiply(dY_hat, dA_prev)

    def predict(self):
        return self.y_hat

    def predictClass(self):
        max_ind = np.argmax(self.y_hat,axis = 0)
        max_prob = np.amax(self.y_hat,axis = 0)
        max_prob_class = [self.classDict[x] for x in max_ind]

        return zip(max_ind,max_prob_class,max_prob)



