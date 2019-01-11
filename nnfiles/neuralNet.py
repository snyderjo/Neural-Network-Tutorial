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
        self.train_loss_vec = []
        self.test_loss_vec = []

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

        count, iterLoss = self.outputL.loss(yIter)

        gradients = self.outputL.backprop(yIter) #loss/output layer gradients
        for lyr in self.layers[-1::-1]: #iterate backwards through hidden layers
            gradients = lyr.backprop(gradients)

        #update parameters
        for lyr in self.layers:
            lyr.update()

        return count, iterLoss


    def epoch(self):
        count, loss = 0, 0
        while not self.feeder.atEnd:
            nwCnt, nwLoss = self.iter()
            count += nwCnt
            loss += nwLoss

        self.feeder.reset()

        return np.divide(loss, count)


    def fit(self,epochCount = 5000, X_new=None, y_new=None):
        trainLossVec = np.zeros(epochCount)
        testLossVec = np.zeros(epochCount)

        count = 0

        while count < epochCount:
            trainLossVec[count] = self.epoch()
            _, testLossVec[count] = self.predict(X_new,y_new)
            count += 1

        self.train_loss_vec.extend(trainLossVec)
        self.test_loss_vec.extend(testLossVec)

        return trainLossVec

    def updateAlpha(self,alpha):
        for lyr in self.layers:
            lyr.updateAlpha(alpha)

    def predict(self,X_new,y):
        activations = X_new
        for lyr in self.layers:
            activations = lyr.forward(activations)
        self.outputL.forward(activations)

        y_hat = self.outputL.predict()
        count, loss = self.outputL.loss(y)

        return y_hat, np.divide(loss,count)

class fullyConnectClassHyper(fullyConnectedClassifier):
    def __init__(self, X, Y, classDict,hiddenLayerSizes, actFuntion = af.relu, miniBatchSize = 64, mutExc = True, alpha =.05, p_keep = 1.0, regular = {"lambd":0, "N":0}, gradClip = np.inf, gradNorm = np.inf):
        self.layers = list()
        self.train_loss_vec = []
        self.test_loss_vec = []

        self.X = X
        self.Y = Y
        self.n_input = np.shape(X)[0]
        self.n_output = np.shape(Y)[0]

        self.feeder = iL.inputLayer(X = X,Y = Y,miniBatchSize = miniBatchSize)

        self.hyperDict = {"alpha":alpha, "p_keep":p_keep, "regular":regular, "gradClip":gradClip, "gradNorm":gradNorm}

        #insert the number of inputs into the hiddenLayerSizes at the 0th position and number of outputs to the final position
        hiddenLayerSizes.insert(0,self.n_input)
        hiddenLayerSizes.append(self.n_output)

        for counter,lyrSize in enumerate(hiddenLayerSizes[1:]):
            self.layers.append(hL.hiddenLayerWHyperparameters(n_self = lyrSize, n_prev = hiddenLayerSizes[counter], name = "hidden" + str(counter), act_func = actFuntion, **self.hyperDict))
        #set output activation layer-specific variables
        self.layers[-1].name = "outputActivations"
        self.layers[-1].p_keep = 1.0 ##make sure to output values for each category.

        if mutExc:
            self.outputL = oL.classMutExcLayer(classDict)
        else:
            self.outputL = oL.classMultOutLayer(classDict)
            #change the activation function of the last hidden layer to linear to allow for negative activations for the sigmoid
            self.layers[-1].activation_func = af.linear

    def iter(self):
        activations, yIter = self.feeder.feed()
        regLoss = 0

        for lyr in self.layers:
            regLoss += np.sum(lyr.regularization_summand())
            activations = lyr.forward(activations)

        self.outputL.forward(activations)

        #sum the batch loss and the regularization loss
        count, iterLoss =  self.outputL.loss(yIter)
        iterLoss += regLoss

        gradients = self.outputL.backprop(yIter) #loss layer gradients
        for lyr in self.layers[-1::-1]: #iterate backwards through hidden layers
            gradients = lyr.backprop(gradients)

        #update parameters
        for lyr in self.layers:
            lyr.update()

        return count, iterLoss

    def predict(self,X_new,y):
        #save p_keep values then change p_keep to 1
        pKeepVal = self.hyperDict["p_keep"]

        self.updateHyperparam(p_keep = 1.0)

        activations = X_new
        regLoss = 0
        for lyr in self.layers:
            regLoss += np.sum(lyr.regularization_summand())
            activations = lyr.forward(activations)
        self.outputL.forward(activations)

        y_hat = self.outputL.predict()
        count, loss = self.outputL.loss(y)
        loss += regLoss

        #return values of p_keep to original values
        self.updateHyperparam(p_keep = pKeepVal)


        return y_hat, np.divide(loss, count)

    def updateHyperparam(self,**kwargs):
        hyperParamNameSet = {"alpha","regular","p_keep","gradClip","gradNorm"}
        updateSet = set(kwargs.keys())
        #filter kwargs to those that are valid

        validDict = {key:kwargs[key] for key in updateSet.intersection(hyperParamNameSet)}
        invalidHyperList = list(updateSet - hyperParamNameSet)

        for lyr in self.layers:
            lyr.updateHyperParams(**validDict)
        if len(invalidHyperList) != 0:
            print("The following are not valid hyperparameter names\n",invalidHyperList)
            print("These are the hyperparameters and their values:")
            while len(hyperParamNameSet) > 0:
                hyperName  = hyperParamNameSet.pop()
                print(hyperName,": ",self.hyperDict[hyperName])

        #make sure dropout does not apply to the final layer
        self.hyperDict.update(validDict)
        self.layers[-1].p_keep = 1.0



