import numpy as np

class relu:

	@staticmethod
	def f(x):
		return np.maximum(np.zeros(x.shape), x)

	@staticmethod
	def f_delta(x):
		return x > 0

class logit:
	@staticmethod
	def f(x):
		return 1 / (1 + np.exp(-x))

	@staticmethod
	def f_delta(x):
		return np.multiply(funtion(x),1 - function(x))

class softPlus:

	@staticmethod
	def f(x):
		return np.log(1 + exp(x))

	@staticmethod
	def f_delta(x):
		return 1 / (1 + np.exp(-x))


class softPlusCentered(softPlus):

	@staticmethod
	def f(x):
		return np.log(1 + exp(x)) - np.log(2)
