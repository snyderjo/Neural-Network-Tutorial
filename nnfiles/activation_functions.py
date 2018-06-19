import numpy as np

class relu:
	@classmethod
	def f(cls,x):
		return np.maximum(np.zeros(x.shape), x)

	@classmethod
	def f_delta(cls,x):
		return x > 0

class logit:
	@classmethod
	def f(cls,x):
		return 1 / (1 + np.exp(-x))

	@classmethod
	def f_delta(cls,x):
		return np.multiply(cls.f(x),1 - cls.f(x))

class softPlus:
	@classmethod
	def f(cls,x):
		return np.log(1 + exp(x))

	@classmethod
	def f_delta(cls,x):
		return 1 / (1 + np.exp(-x))


class softPlusCentered(softPlus):
	@classmethod
	def f(cls,x):
		return np.log(1 + exp(x)) - np.log(2)

class linear:
	@classmethod
	def f(cls,x):
		return x

	@classmethod
	def f_delta(cls,x):
		return 1