import numpy as np
from activation_functions import *



class baseHiddenLayer:

	def __init__(self, n_self, n_prev, name, act_func = relu, alpha = .05):
		self.name = name
		self.n_nodes = n_self
		self.activation_func = act_func
		self.alpha = alpha

		self.W = np.divide(np.random.randn(n_self,n_prev).astype(np.float128) , np.sqrt(n_prev))
		self.b = np.zeros((n_self,1),dtype = np.float128)

		self.dW = np.zeros(self.W.shape, dtype = np.float128)
		self.db = np.zeros(self.b.shape, dtype = np.float128)

		self.A_prev = np.zeros([], dtype = np.float128)
		self.Z = np.zeros([], dtype = np.float128)

	def forward(self,A_prev):
		self.A_prev = A_prev
		self.Z = np.add(np.matmul(self.W,A_prev) , self.b)

		return self.activation_func.f(self.Z)

	def backprop(self,dA_back):
		dZ = np.multiply(dA_back, self.activation_func.f_delta(self.Z))
		m = dZ.shape[1]

		 #matmul has no means of dealing with NAN's, i.e. may have to re-write the below.
		self.dW = (1/m) * np.matmul(dZ, np.transpose(self.A_prev))
		self.db = (1/m) * np.matmul(dZ, np.ones([m,1]))

		dA_prev = np.matmul(np.transpose(self.W), dZ)

		return dA_prev

	def update(self):
		self.W = np.subtract(self.W , np.multiply(self.alpha , self.dW))
		self.b = np.subtract(self.b , np.multiply(self.alpha , self.db))

		#clear the gradients?

	def updateAlpha(self,newAlpha):
		self.alpha = newAlpha


class hiddenLayerDropOut(baseHiddenLayer):
	def __init__(self,p_drop = .5):
		self.p_drop = p_drop

