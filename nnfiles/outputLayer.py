import numpy as np
from abc import ABC, abstractmethod
from activation_functions import logit


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

    @abstractmethod
    def predictClass(self):
        pass

    def update(self):
        pass


class classMultOutLayer(baseOuputLayerClassifier):
    ##needs a non-relu layer immediately before to enable negative activations
    def __init__(self,classDict):
        self.classDict = classDict
        self.y_hat = None
        self.a_prev = None

    def forward(self,a_prev):
        self.y_hat = logit.f(a_prev)

    def loss(self,y):
        losses = - np.multiply(y, np.log(self.y_hat)) - np.multiply(1 - y, np.log(1 - self.y_hat))

        return np.mean(losses)

    def backprop(self,y):
        loss_grad = self.y_hat - y
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
        self.y_hat = None
        self.a_prev = None

    @classmethod
    def softmax(cls,act_matrix):
        max_a = np.amax(act_matrix,axis = 0)
        exp_a = np.exp(act_matrix - max_a)
        sum_exp_a = np.sum(exp_a,0)

        return np.divide(exp_a,sum_exp_a)

    @classmethod
    def softmax_delta(cls,act_matrix,y):
        exp_a = np.exp(act_matrix)
        sum_exp_a = np.sum(exp_a,0)

        #the gradient in the event that the max(y_hat) is y
        numerator_one = np.multiply(act_matrix, np.multiply(exp_a, (sum_exp_a - exp_a)))
        denominator_one = np.square(sum_exp_a)

        #the gradient in the event that the max(y_hat) is not y
        #need to take what the numerator of the y_hat eventually became, i.e. that of y
        softmax_numerator = np.max(np.multiply(exp_a, y) ,axis = 0) #exp(a) is guaranteed to be positive, and the non-numerator values will be zero
        numerator_zero = -np.multiply(softmax_numerator,np.multiply(act_matrix,exp_a))
        denominator_zero = np.square(sum_exp_a)

        soft_delta  = np.multiply(y,np.divide(numerator_one,denominator_one)) + np.multiply(1 - y,np.divide(numerator_zero,denominator_zero))

        return soft_delta


    def forward(self,a_prev):
        self.a_prev = a_prev
        self.y_hat = self.softmax(a_prev)

    def loss(self,y):
        """
        Assumes that y is a one-hot matrix of dimensions nClass x m
        """
        losses = -np.multiply(y, np.log(self.y_hat))

        return np.mean(losses)

    def backprop(self,y):
        dY_hat = np.divide(y,self.y_hat) #mostly zeros

        dA_prev = self.softmax_delta(self.a_prev,y)

        return np.multiply(dY_hat, dA_prev)

    def predict(self):
        return self.y_hat

    def predictClass(self):
        max_ind = np.argmax(self.y_hat,axis = 0)
        max_prob = np.amax(self.y_hat,axis = 0)
        max_prob_class = [self.classDict[x] for x in max_ind]

        return zip(max_ind,max_prob_class,max_prob)



