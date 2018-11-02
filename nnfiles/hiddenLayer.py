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

		self.__dict__.update(kwargs)

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
	def __init__(self,hyperparams = {"alpha":.05, "p_drop":0.0, "lambd":0, "regN":2, "gradThresh" = np.inf, "gradClip": = np.inf}):
		self.alpha = hyperparams["alpha"]
		self.lambd = hyperparams["lambd"]
		self.regN = hyperparams["regN"]
		self.p_drop = hyperparams["p_drop"]
		self.gradThresh = hyperparams["gradThresh"]

	def reglatrization_summand(self):
		regLossSummand = np.zeros(*self.W.shape, dtype = np.float128)
		if self.regN == 1:
			regLossSummand = np.multiply(self.lambd,np.abs(self.W))

		elif self.regN == 2:
			regLossSummand = np.multiply(self.lambd,np.square(self.W))

		else:
			raise ValueError

		return regLossSummand

	def regulatrization_summand_delta(self):
		dRegLossSummand = np.zeros(*self.W.shape, dtype = np.float128)

		if self.regN == 1:
			cond = self.W > 0
			dRegLossSummand = np.sum(1*cond, -1*(np.logical_not(cond)))

		elif self.regN == 2:
			dRegLossSummand = np.multiply(self.lambd,self.W)

		else:
			raise ValueError

		return dRegLossSummand

	def update(self):
		self.clipGrads()

		self.W = np.subtract(self.W , np.multiply(self.alpha , np.sum(self.dW, self.regulatrization_summand_delta())))
		self.b = np.subtract(self.b , np.multiply(self.alpha , self.db))


	def clipGrads(self):
		l2_norm = np.sum(np.square(self.dW))
		if l2_norm > self.gradThresh:
			self.dW *= np.divide(self.gradThresh,l2_norm)

	def forward(self,A_prev):
		self.A_prev = A_prev
		self.Z = np.add(np.matmul(self.W,A_prev) , self.b)

		ActMat = self.activation_func.f(self.Z)

		U = np.divide(np.random.rand(*ActMat.shape) > self.p_drop),self.p_drop)

		ActMat *= U

		return ActMat





