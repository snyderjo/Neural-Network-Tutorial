import numpy as np
import inputLayer as iL
import hiddenLayer as hL
import outputLayer as oL
import activation_functions as af


class fullyConnectedClassifier():
    def __init__(self, X, Y, classDict,hiddenLayerSizes, actFuntion = af.relu, miniBatchSize = 64, mutExc = True, alpha = .05):
        self.layers = list() #list or dict?
        self.X = X
        self.Y = Y
        self.n_input = np.shape(X)[0]
        self.n_output = np.shape(Y)[0]
        self.feeder = iL.inputLayer(X = X,Y = Y,miniBatchSize = miniBatchSize)
        self.loss_vec = []

        #insert the number of input and output nodes in the appropriate places
        hiddenLayerSizes.insert(0,self.n_input)
        hiddenLayerSizes.append(self.n_output)

        self.layers.append(hL.baseHiddenLayer(n_self = hiddenLayerSizes[0], n_prev = self.n_input, name = "hidden" + str(0), act_func = actFuntion, alpha = alpha))
        for counter,lyrSize in enumerate(hiddenLayerSizes[1:]):
            self.layers.append(hL.baseHiddenLayer(n_self = lyrSize, n_prev = self.layers[-1].n_nodes, name = "hidden" + str(counter), act_func = actFuntion, alpha = alpha))

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

        iterLoss = self.layers[-1].loss(yIter)

        gradients = self.layers[-1].backprop(yIter) #loss layer gradients
        for lyr in self.layers[-2::-1]: #iterate backwards through hidden layers
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

class fullyConnectClassHyper(fullyConnectedClassifier):
    def __init__(self, X, Y, classDict,hiddenLayerSizes, actFuntion = af.relu, miniBatchSize = 64, mutExc = True, **kwargs):
        self.layers = list() #list or dict?
        self.loss_vec = []

        self.X = X
        self.Y = Y
        self.n_input = np.shape(X)[0]
        self.n_output = np.shape(Y)[0]

        self.feeder = iL.inputLayer(X = X,Y = Y,miniBatchSize = miniBatchSize)

        self.__dict__.update(kwargs)

        #insert the number of inputs into the hiddenLayerSizes at the 0th position and number of outputs to the final position
        hiddenLayerSizes.insert(0,self.n_input)
        hiddenLayerSizes.append(self.n_output)

        for counter,lyrSize in enumerate(hiddenLayerSizes[1:]):
            self.layers.append(hL.baseHiddenLayer(n_self = lyrSize, n_prev = hiddenLayerSizes[counter], name = "hidden" + str(counter), act_func = actFuntion, kwargs))


        self.layers[-1].name = "outputActivations"

        if mutExc:
            self.outputL = oL.classMutExcLayer(classDict)
        else:
            self.outputL = oL.classMultOutLayer(classDict)
            #change the activation function of the last hidden layer to linear to allow for negative activations
            self.layers[-1].activation_func = af.linear

    def iter(self):
        activations, yIter = self.feeder.feed()
        regLoss = 0

        for lyr in self.layers:
            #add regularization loss
            activations = lyr.forward(activations)

        self.outputL.forward(activations)

        iterLoss = self.outputL.loss(yIter)

        gradients = self.outputL.backprop(yIter) #loss layer gradients
        for lyr in self.layers[-1::-1]: #iterate backwards through hidden layers
            gradients = lyr.backprop(gradients)

        #update parameters
        #map(lambda x: x.update(), self.layers) #test this!!!  DOES NOT WORK
        for lyr in self.layers:
            lyr.update()

        return iterLoss



