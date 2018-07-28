import numpy as np
from activation_functions import *



class baseHiddenLayer:

	def __init__(self, n_self, n_prev, act_func = relu, alpha = .05):
		self.n_nodes = n_self
		self.activation_func = act_func
		self.alpha = alpha

		self.W = np.random.randn(n_self,n_prev) / np.sqrt(n_prev)
		self.b = np.zeros((n_self,1))

		self.dW = np.zeros(self.W.shape)
		self.db = np.zeros(self.b.shape)

		self.A_prev = np.zeros([], dtype = np.float64)
		self.Z = np.zeros([], dtype = np.float64)

	def forward(self,A_prev):
		self.A_prev = A_prev
		self.Z = np.matmul(self.W,A_prev) + self.b

		return self.activation_func.f(self.Z)

	def backprop(self,dA_back):
		dZ = np.multiply(dA_back,self.activation_func.f_delta(self.Z))
		m = dZ.shape[0]

		self.dW = (1/m) * np.matmul(dZ, np.transpose(self.A_prev))
		self.db = (1/m) * np.multiply(dZ, np.ones([self.n_nodes,1]))

		dA_prev = np.matmul(np.transpose(self.W), dZ)

		return dA_prev

	def update(self):
		self.W -= self.alpha * self.dW
		self.b -= self.alpha * self.db

		#clear the gradients?


class hiddenLayerDropOut(baseHiddenLayer):
	def __init__(self,p_drop = .5):
		self.p_drop = p_drop

