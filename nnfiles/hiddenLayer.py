import numpy as np
from activation_functions import *



class baseHiddenLayer:

	def __init__(self, n_self, n_prev, name, act_func = relu,**kwargs):
		self.name = name
		self.n_nodes = n_self
		self.activation_func = act_func

		self.W = np.divide(np.random.randn(n_self,n_prev).astype(np.float128) , np.sqrt(n_prev))
		self.b = np.zeros((n_self,1),dtype = np.float128)

		self.dW = np.zeros(self.W.shape, dtype = np.float128)
		self.db = np.zeros(self.b.shape, dtype = np.float128)

		self.A_prev = np.zeros([], dtype = np.float128)
		self.Z = np.zeros([], dtype = np.float128)

		self.__dict__.update(kwargs) #for alpha

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


class hiddenLayerWHyperparameters(baseHiddenLayer):
	def __init__(self,n_self, n_prev, name, act_func = relu, **kwargs):
		self.name = name
		self.n_nodes = n_self
		self.activation_func = act_func

		self.W = np.divide(np.random.randn(n_self,n_prev).astype(np.float128) , np.sqrt(n_prev))
		self.b = np.zeros((n_self,1),dtype = np.float128)

		self.dW = np.zeros(self.W.shape, dtype = np.float128)
		self.db = np.zeros(self.b.shape, dtype = np.float128)

		self.A_prev = np.zeros([], dtype = np.float128)
		self.Z = np.zeros([], dtype = np.float128)

		self.alpha =.05
		self.p_keep = 1.0
		self.regular = {"lambd":0, "N":0}
		self.gradThresh = np.inf

		self.__dict__.update(kwargs) #for hyperparameter updates


	def regularization_summand(self):
		regLossSummand = np.zeros(self.W.shape, dtype = np.float128)

		if self.regular["N"] == 0:
			pass

		elif self.regular["N"] == 1:
			regLossSummand = np.multiply(self.regular["lambd"],np.abs(self.W))

		elif self.regular["N"] == 2:
			regLossSummand = np.divide(np.multiply(self.regular["lambd"], np.square(self.W)), 2)

		else:
			raise ValueError

		return regLossSummand

	def regularization_summand_delta(self):
		dRegLossSummand = np.zeros(self.W.shape, dtype = np.float128)

		if self.regular["N"] == 0:
			pass

		elif self.regular["N"] == 1:
			cond = self.W > 0
			dRegLossSummand = np.multiply(self.regular["lambd"], np.sum(1*cond, -1*(np.logical_not(cond))))

		elif self.regular["N"] == 2:
			dRegLossSummand = np.multiply(self.regular["lambd"], self.W)

		else:
			raise ValueError

		return dRegLossSummand

	def update(self):
		self.clipGrads()

		self.W = np.subtract(self.W, np.multiply(self.alpha , np.add(self.dW, self.regularization_summand_delta())))
		self.b = np.subtract(self.b, np.multiply(self.alpha , self.db))

	def updateHyperParams(self,**kwargs):
		self.__dict__.update(kwargs)

	def clipGrads(self):
		self.dW[self.dW > self.gradClip] = self.gradClip

		l2_norm = np.sqrt(np.sum(np.square(self.dW)))
		if l2_norm > self.gradNorm:
			self.dW *= np.divide(self.gradNorm,l2_norm)

	def forward(self,A_prev):
		self.A_prev = A_prev
		self.Z = np.add(np.matmul(self.W,A_prev) , self.b)

		ActMat = self.activation_func.f(self.Z)

		U = np.divide((np.random.rand(*ActMat.shape) < self.p_keep),self.p_keep)

		ActMat *= U

		return ActMat
