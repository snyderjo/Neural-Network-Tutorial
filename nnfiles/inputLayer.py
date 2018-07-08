import numpy as np

class inputLayer():
    def __init__(self, data, subSize = 64):
        self.dataset = data #data should be of the dimentions n_variables x n_obs
        self.subSize = subSize
        self.m = np.shape(data)[1]
        self.indexOrder = np.random.shuffle(np.arange(self.m))
        self.start = 0
        self.atEnd = False

    def feed(self):
        end = self.start + self.subSize
        if end < self.m:
            iterData = self.dataset[:,self.indexOrder[self.start,end]]
            self.start += self.subSize
        else:
            iterData = self.dataset[:,self.indexOrder[self.start,self.m]]
            self.atEnd = True

        return iterData

    def reset(self):
        np.random.shuffle(self.indexOrder)
        self.start = 0
        self.atEnd = False
