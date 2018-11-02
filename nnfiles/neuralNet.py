import numpy as np
import inputLayer as iL
import hiddenLayer as hL
import outputLayer as oL
import activation_functions as af


class fullyConnectedClassifier():
    def __init__(self, X, Y, classDict,hiddenLayerSizes, actFuntion = af.relu, alpha = .05, miniBatchSize = 64, mutExc = True):
        self.layers = list() #list or dict?
        self.X = X
        self.Y = Y
        self.n_input = np.shape(X)[0]
        self.n_output = np.shape(Y)[0]
        self.feeder = iL.inputLayer(X = X,Y = Y,miniBatchSize = miniBatchSize)
        self.loss_vec = []

        #insert the input and output nodes in the appropriate places
        hiddenLayerSizes.insert(0,self.n_input)
        hiddenLayerSizes.append(self.n_output)

        for counter,lyrSize in enumerate(hiddenLayerSizes[1:]):
            self.layers.append(hL.baseHiddenLayer(n_self = lyrSize, n_prev = hiddenLayerSizes[counter], name = "hidden" + str(counter), act_func = actFuntion, alpha = alpha))
        self.layers[-1].name = "outputActivations"

        if mutExc:
            self.outputL = oL.classMutExcLayer(classDict)
        else:
            self.outputL = oL.classMultOutLayer(classDict)
            #change the activation function of the last hidden layer to linear to allow for negative activations
            self.layers[-1].activation_func = af.linear


    def iter(self):
        activations, yIter = self.feeder.feed()
        for lyr in self.layers:
            activations = lyr.forward(activations)

        self.outputL.forward(activations)

        iterLoss = self.outputL.loss(yIter)

        gradients = self.outputL.backprop(yIter) #loss/output layer gradients
        for lyr in self.layers[-1::-1]: #iterate backwards through hidden layers
            gradients = lyr.backprop(gradients)


        #update parameters
        #map(lambda x: x.update(), self.layers) #test this!!!  DOES NOT WORK
        for lyr in self.layers:
            lyr.update()

        return iterLoss


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

        self.loss_vec.extend(lossVec)

        return lossVec

    def updateAlpha(self,alpha):
        for lyr in self.layers:
            lyr.updateAlpha(alpha)

    def predict(self,X_new):
        #very inefficient in terms of memory
        activations = X_new
        for lyr in self.layers:
            activations = lyr.forward(activations)

        self.outputL.forward(activations)

        return self.outputL.predict()

    def predict(self,X_new,y):
        activations = X_new
        for lyr in self.layers:
            activations = lyr.forward(activations)

        y_hat = self.outputL.predict()
        loss = self.outputL.loss(y)

        return y_hat, loss


