import numpy as np
from abc import ABC, abstractmethod
from nnfiles.activation_functions import logit

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

class baseOuputLayerRegresser(ABC):
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
    def update(self):
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
            # = - y * log(y_hat) - (1 - y) * log(1 - y_hat)

        return losses.shape[1], np.sum(losses)

    def backprop(self,y):
        loss_grad = np.subtract(self.y_hat , y)
        return loss_grad

    def predict(self):
        return self.y_hat

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
    def softmax_d(cls,actVec):
        maxA = np.amax(actVec)
        expA = np.exp(np.subtract(actVec, maxA))
        sumExpA = np.sum(expA)

        numer = -np.outer(expA,expA) #off-diagonal gradient

        diag_num = np.multiply(expA,np.subtract(sumExpA, expA)) #diagonal gradient

        np.fill_diagonal(numer,diag_num)

        return np.divide(numer, np.square(sumExpA))


    def forward(self,A_prev):
        self.A_prev = A_prev
        self.y_hat = self.softmax(A_prev)


    def loss(self,y):
        """
        Assumes that y is a one-hot matrix of dimensions nClass x m
        """
        losses = -np.multiply(y, np.log(self.y_hat))

        return losses.shape[1], np.sum(losses)

    def backprop(self,y):
        #avoid dividing by near-zero y_hat values
        dY_hat = np.zeros(self.y_hat.shape,dtype = np.float128)
        dY_hat[y == 1] = np.divide(-1, self.y_hat[y == 1]) #y_hat's should be larger for y == 1 -- fewer near-zero divisors

        nClass, m = dY_hat.shape

        #transpose the activation matrix, and apply softmax_d to create a stack of matrices--necessary for np.matmul
        transpA_prev = self.A_prev.transpose()
        deltaSoftMats = np.apply_along_axis(self.softmax_d,1,transpA_prev)

        #transpose dYhat and turn it into a stack of matrices of dim (nClass, 1) rather than vectors of dim (nClass,)
        transpDY_hat = dY_hat.transpose()
        transpDY_hat_mats = np.apply_along_axis(np.reshape,1,transpDY_hat,(nClass,1))

        #multiply the matrices reshape back to a stack of vectors, and transpose back
        dA_prevDY_hat = np.matmul(deltaSoftMats,transpDY_hat_mats).reshape(m,nClass).transpose()

        return dA_prevDY_hat

    def predict(self):
        return self.y_hat

    def predictClass(self):
        max_ind = np.argmax(self.y_hat,axis = 0)
        max_prob = np.amax(self.y_hat,axis = 0)
        max_prob_class = [self.classDict[x] for x in max_ind]

        return zip(max_ind,max_prob_class,max_prob)


class regressionOutputLayer(baseOuputLayerRegresser):
    def __init__(self,n_act):
        self.y_hat = np.zeros
        self.W = np.zeros()
        self.b = np.zeros((1,1))

    def forward(self,A_prev):
        self.y_hat = A_prev

    def loss(self,y):
        return np.mean(np.square(np.subtract(self.y_hat, y)))

    def backprop(self,y):
        return np.multiply(2,np.subtract(self.y_hat, y))

    def predict(self,A_prev):
        return A_prev
