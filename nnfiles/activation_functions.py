import numpy as np

class relu:

	def function(self,x):
		return np.max(0,x)
		
	def function_delta(self,x):
		return x > 0
		
class logit:

	def function(self,x):
		return 1 / (1 + np.exp(-x))
		
	def function_delta(self,x):
		return np.mult(self.funtion(x),1 - self.function(x))

class softPlus:

	def function(self,x):
		return np.log(1 + exp(x))
		
	def function_delta(self,x):
		return 1 / (1 + np.exp(-x))


class softPlusCentered(softMax):

	def function(self,x):
		return np.log(1 + exp(x)) - np.log(2)
			