import numpy as np

class inputLayer():
    def __init__(self, X, Y, miniBatchSize = 64):
        self.X = X #data should be of the dimentions n_variables x n_obs
        self.miniBatchSize = miniBatchSize
        self.m = np.shape(X)[1]
        self.indexOrder = np.arange(self.m)
        np.random.shuffle(self.indexOrder)
        self.start = 0
        self.atEnd = False
        self.Y = Y

    def feed(self):
        """
        returns minibatches for X and Y respectively
        """
        end = self.start + self.miniBatchSize

        if end < self.m:
            iterXData = self.X[:,self.indexOrder[self.start:end]]
            iterYData = self.Y[:,self.indexOrder[self.start:end]]
            self.start += self.miniBatchSize
        else:
            iterXData = self.X[:,self.indexOrder[self.start:self.m]]
            iterYData = self.Y[:,self.indexOrder[self.start:self.m]]
            #update self.start???  nahh
            self.atEnd = True

        return iterXData, iterYData

    def reset(self):
        """
        resets data for another epoch
        """
        np.random.shuffle(self.indexOrder)
        self.start = 0
        self.atEnd = False
