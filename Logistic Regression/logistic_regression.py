import pandas as pd
import numpy as np

# Sigmoid (Logistic) Function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

class Logistic_Regression:

	def __init__(self, data, feature_size):

		self.data = data
		self.feature_size = feature_size
		self.weights = np.zeros(self.feature_size,dtype=float)
		self.bias = 0
		self.batch_size = len(data)


	def classifier(self, instance):
		# Not a probability but a number
		# y = w.x + b 
		return np.dot(self.weights,instance[0]) + self.bias

	def predict(self, instance):
		P = sigmoid(self.classifier(instance))
		if (P > 0.5):
			return 1
		return 0

	def cross_entropy_loss(self):
		cost = 0
		for instance in self.data:
			cost += (self.predict(instance) * np.log(sigmoid(self.classifier(instance))) + ((1-instance[1])*np.log(1-sigmoid(self.classifier(instance)))))

		return cost / self.batch_size

	## Gradient Descent

	def Batch_Graident_Descent(self, epoch=10000, batch_size=75, learning_rate=0.001):

		weights = np.zeros(self.feature_size,dtype=float) # [0,0,0,0]
		bias = 0
		# Cost of 
		costs = []

		while epoch > 0:

			weight_0_loss = weight_1_loss = weight_2_loss = weight_3_loss = bias_loss = 0

			for instance in self.data[:batch_size]:

				feature_vector = instance[0]
				target = instance[1]

				weight_0_loss += (sigmoid(self.classifier(feature_vector))-instance[1]) * instance[0][0]
				weight_1_loss += (sigmoid(self.classifier(feature_vector))-instance[1]) * instance[0][1]
				weight_2_loss += (sigmoid(self.classifier(feature_vector))-instance[1]) * instance[0][2]
				weight_3_loss += (sigmoid(self.classifier(feature_vector))-instance[1]) * instance[0][3]

				bias_loss +=  sigmoid(self.classifier(instance)) - instance[1]

			self.weights[0] -= learning_rate * (weight_0_loss / batch_size)
			self.weights[1] -= learning_rate * (weight_1_loss / batch_size)
			self.weights[2] -= learning_rate * (weight_2_loss / batch_size)
			self.weights[3] -= learning_rate * (weight_3_loss / batch_size)

			self.bias -= learning_rate * (sigmoid(self.classifier(instance)) - instance[1])

			costs.append(self.cross_entropy_loss())

			epoch -= 1

		return self.weights, self.bias, costs

	def Stochastic_Graident_Descent(self, learning_rate=0.1):

		costs = []

		for instance in self.data:

			self.weights[0] -= learning_rate * (sigmoid(self.classifier(instance))-instance[1])*instance[0][0]
			self.weights[1] -= learning_rate * (sigmoid(self.classifier(instance))-instance[1])*instance[0][1]
			self.weights[2] -= learning_rate * (sigmoid(self.classifier(instance))-instance[1])*instance[0][2]
			self.weights[3] -= learning_rate * (sigmoid(self.classifier(instance))-instance[1])*instance[0][3]

			self.bias -= learning_rate * (sigmoid(self.classifier(instance)) * instance[1])

			costs.append(self.cross_entropy_loss())

		return self.weights, self.bias, costs



# Sigmoid (Logistic) Function
def softmax(x):
	return np.exp(x) / (np.sum(lambda x: np.exp(x)))

class Multinomial_Logistic_Regression:

	def __init__(self, data, feature_size, target):

		self.data = data
		self.feature_size = feature_size
		self.target = target
		
		self.weights = np.zeros((self.feature_size, self.target),dtype=float)
		self.bias = np.zeros((self.target), dtype=float)
		self.batch_size = len(data)


	def classifier(self, instance):
		# Not a probability but a number
		# y = w.x + b 
		return np.dot(self.weights,instance[0]) + self.bias

	def predict(self, instance):
		P = sigmoid(self.classifier(instance))
		if (P > 0.5):
			return 1
		return 0

	def cross_entropy_loss(self):
		cost = 0
		for instance in self.data:
			cost += (self.predict(instance) * np.log(sigmoid(self.classifier(instance))) + ((1-instance[1])*np.log(1-sigmoid(self.classifier(instance)))))

		return cost / self.batch_size

	## Gradient Descent

	def Batch_Graident_Descent(self, epoch=10000, batch_size=75, learning_rate=0.001):

		weights = np.zeros(self.feature_size,dtype=float) # [0,0,0,0]
		bias = 0
		# Cost of 
		costs = []

		while epoch > 0:

			weight_0_loss = weight_1_loss = weight_2_loss = weight_3_loss = bias_loss = 0

			for instance in self.data[:batch_size]:

				feature_vector = instance[0]
				target = instance[1]

				weight_0_loss += (sigmoid(self.classifier(feature_vector))-instance[1]) * instance[0][0]
				weight_1_loss += (sigmoid(self.classifier(feature_vector))-instance[1]) * instance[0][1]
				weight_2_loss += (sigmoid(self.classifier(feature_vector))-instance[1]) * instance[0][2]
				weight_3_loss += (sigmoid(self.classifier(feature_vector))-instance[1]) * instance[0][3]

				bias_loss +=  sigmoid(self.classifier(instance)) - instance[1]

			self.weights[0] -= learning_rate * (weight_0_loss / batch_size)
			self.weights[1] -= learning_rate * (weight_1_loss / batch_size)
			self.weights[2] -= learning_rate * (weight_2_loss / batch_size)
			self.weights[3] -= learning_rate * (weight_3_loss / batch_size)

			self.bias -= learning_rate * (sigmoid(self.classifier(instance)) - instance[1])

			costs.append(self.cross_entropy_loss())

			epoch -= 1

		return self.weights, self.bias, costs

	def Stochastic_Graident_Descent(self, learning_rate=0.1):

		costs = []

		for instance in self.data:

			self.weights[0] -= learning_rate * (sigmoid(self.classifier(instance))-instance[1])*instance[0][0]
			self.weights[1] -= learning_rate * (sigmoid(self.classifier(instance))-instance[1])*instance[0][1]
			self.weights[2] -= learning_rate * (sigmoid(self.classifier(instance))-instance[1])*instance[0][2]
			self.weights[3] -= learning_rate * (sigmoid(self.classifier(instance))-instance[1])*instance[0][3]

			self.bias -= learning_rate * (sigmoid(self.classifier(instance)) * instance[1])

			costs.append(self.cross_entropy_loss())

		return self.weights, self.bias, costs
