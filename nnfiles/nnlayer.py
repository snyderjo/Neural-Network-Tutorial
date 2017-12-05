import numpy as np
from activation_functions import *



class baseHiddenLayer:
	
	def __init__(self, n_self, n_prev, act_func = relu):
		self.n_nodes = n_self
		self.activation_func = act_func

		self.W = np.zeros((n_self,n_prev), type = np.float32)
		self.b = np.zeros((n_self,1), type = np.float32)
		
		self.dW = np.zeros(W.shape, type = np.float32)
		self.db = np.zeros(b.shape, type = np.float32)
		
		self.Z = np.zeros([],ndmin = 2, type = np.float32)
		
	def forward(self,A_prev):
		self.Z = np.matmult(A_prev,self.W) + self.b
		
		return activation_func.function(self.Z)
	
	def backprop(self,dA_back):
	
		dZ = np.matmult(dA_back,act_func_delta(self.Z))
		m = dZ.shape[1]
		
		self.dW = (1/m) * np.matmult(dZ, np.transpose(activation_func.function_delta(self.Z)))
		self.db = (1/m) * np.matmult(dZ, np.ones(n_nodes,1))
		
		dA_prev = np.matmult(transpose(self.W), self.dZ)
		
		return self.dA_prev
	
	
	def update(self,alpha):
		self.W -= alpha * self.dW
		self.b -= alpha * self.db
		
	def initialize(self):
		pass
		
		
		
class hiddenLayerDropOut(baseHiddenLayer):
	def __init__(self,p_drop = .5):
		self.p_drop = p_drop
		
		