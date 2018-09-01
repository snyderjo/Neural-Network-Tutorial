import numpy as np
import inputLayer as iL
import hiddenLayer as hL
import outputLayer as oL
import activation_functions as af


class fullyConnectedClassifier():
    def __init__(self, X, Y, hiddenLayerSizes, actFuntion = af.relu, alpha = .05, miniBatchSize = 64, mutExc = True):
        self.layers = list() #list or dict?
        self.X = X
        self.Y = Y
        self.n_input = np.shape(X)[0]
        self.n_output = np.shape(Y)[0]
        self.feeder = iL.inputLayer(X = X,Y = Y,miniBatchSize = miniBatchSize)

        self.layers.append(hL.baseHiddenLayer(n_self = hiddenLayerSizes[0], n_prev = self.n_input, act_func = actFuntion, alpha = alpha))
        for lyrSize in hiddenLayerSizes[1:]:
            self.layers.append(hL.baseHiddenLayer(n_self = lyrSize, n_prev = self.layers[-1].n_nodes, act_func = actFuntion, alpha = alpha))
        self.layers.append(hL.baseHiddenLayer(n_self = self.n_output, n_prev = hiddenLayerSizes[-1], act_func = actFuntion, alpha = alpha))

        if mutExc:
            self.layers.append(oL.classMutExcLayer())
        else:
            self.layers.append(oL.classMultOutLayer())
            #change the activation function of the last hidden layer to linear to allow for negative activations
            self.layers[-2].activation_func = af.linear


    def iter(self):
        activations, yIter = self.feeder.feed()
        for lyr in self.layers:
            activations = lyr.forward(activations)

        gradients = self.layers[-1].backprop(yIter) #loss layer gradients
        for lyr in self.layers[-2::-1]: #iterate backwards through hidden layers
            print(lyr)
            print(gradients)
            print(lyr.Z)
            gradients = lyr.backprop(gradients)

        #update parameters
        map(lambda x: x.update(), self.layers) #test this!!!

        return self.layers[-1].loss(yIter)


    def epoch(self):
        loss = 0
        while not self.feeder.atEnd:
            loss += self.iter()

        self.feeder.reset()

        return loss


    def fit(self,epochCount = 5000):
        lossVec = np.zeros(epochCount)

        count = 0

        while count < epochCount:
            lossVec[count] = self.epoch()
            count += 1

        return lossVec

    def updateAlpha(self,alpha):
        for lyr in self.layers:
            lyr.updateAlpha(alpha)

    def predict(self,X_new):
        activations = X_new
        for lyr in self.layers:
            activations = lyr.forward(activations)

        return self.layers[-1].predict()

    def predict(self,X_new,y):
        activations = X_new
        for lyr in self.layers:
            activations = lyr.forward(activations)

        y_hat = self.layers[-1].predict()
        loss = self.layers[-1].loss(y)

        return y_hat, loss


