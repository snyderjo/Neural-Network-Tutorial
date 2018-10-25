import numpy as np
from abc import ABC, abstractmethod

class baseActFunction(ABC):
	@classmethod
	@abstractmethod
	def f(cls,x):
		pass

	@classmethod
	@abstractmethod
	def f_delta(cls,x):
		pass

class relu(baseActFunction):
	def f(x):
		return np.maximum(np.zeros(x.shape), x)

	def f_delta(x):
		return x > 0

class logit(baseActFunction):
	def f(x):
		return np.divide(1 , np.add(1 , np.exp(-x)))

	@classmethod
	def f_delta(cls,x):
		return np.multiply(cls.f(x),np.subract(1 , cls.f(x)))

class softPlus(baseActFunction):
	def f(x):
		return np.log(np.add(1 , exp(x)))

	def f_delta(x):
		return np.divide(1 , (np.add(1 , np.exp(-x))))

class softPlusCentered(softPlus):
	def f(x):
		return np.log(np.add(1 , exp(x))) - np.log(2)

class linear(baseActFunction):
	def f(x):
		return x

	def f_delta(x):
		return 1