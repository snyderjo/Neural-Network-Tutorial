import numpy as np
from activation_functions import *



class baseHiddenLayer:

	def __init__(self, n_self, n_prev, act_func = relu, alpha = .05):
		self.n_nodes = n_self
		self.activation_func = act_func
		self.alpha = alpha

		self.W = np.random.randn(n_self, n_prev) / np.sqrt(n_prev)
		self.b = np.zeros((n_self,1), type = np.float32)

		self.dW = np.zeros(W.shape, type = np.float32)
		self.db = np.zeros(b.shape, type = np.float32)

		self.A_prev = np.zeros([], type = np.float32)
		self.Z = np.zeros([],ndmin = 2, type = np.float32)

	def forward(self,A_prev):
		self.A_prev = A_prev
		self.Z = np.matmult(A_prev,self.W) + self.b

		return self.activation_func.f(self.Z)

	def backprop(self,dA_back):
		dZ = np.multiply(dA_back,self.activation_func.f_delta(self.Z))
		m = dZ.shape[0]

		self.dW = (1/m) * np.matmult(dZ, np.transpose(self.A_prev))
		self.db = (1/m) * np.multiply(dZ, np.ones(self.n_nodes,1))

		dA_prev = np.matmult(transpose(self.W), self.dZ)

		return dA_prev

	def update(self,alpha):
		self.W -= alpha * self.dW
		self.b -= alpha * self.db

		#clear the gradients?


class hiddenLayerDropOut(baseHiddenLayer):
	def __init__(self,p_drop = .5):
		self.p_drop = p_drop

