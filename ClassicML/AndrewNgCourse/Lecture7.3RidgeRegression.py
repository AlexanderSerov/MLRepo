import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import minmax_scale


class MyRidgeRegression:

	theta_parameters = None

	def __init__(self, alpha=0):

		self.alpha = alpha

	def fit(self, X, y):

		self.GradientDescent(X, y)

	def GradientDescent(self, X, y):
		
		X = minmax_scale(X, feature_range=(0, 1))

		ones = np.ones((X.shape[0],1))
		X = np.concatenate((ones,X), axis=1)

		theta_vector = np.zeros(X.shape[1])     #Fill vector with zero values
		learning_rate = 0.1
		theta_temp = 0

		for i in range(10000):
			theta_derivatives = (1/len(X))*np.sum(X.T * (X @ theta_vector - y), axis=1)
			theta_vector = theta_vector*(1 - learning_rate*(self.alpha/len(X))) - learning_rate * theta_derivatives		# Regularization term.

			for idx, item in enumerate(theta_vector):
				print(" theta_", idx, " = ", item, sep='', end='')
			print('')

			if (math.isclose(theta_vector[1],theta_temp, rel_tol=1e-8)):
				print("Break due to isclose function")
				break
			elif (abs(theta_vector[1]) > sys.maxsize):
				print("Break due to error function exceed")
				break

			theta_temp = theta_vector[0]
					
		self.theta_parameters = theta_vector

	@staticmethod
	def MeanNormalization(X):

		Range = np.max(X) - np.min(X)      #Range in data set

		return (X - np.min(X)) / Range


def DrawPlot(X,y):

	X = minmax_scale(X, feature_range=(0, 1))
	DotsForSmoothLine = np.linspace(np.amin(X), np.amax(X), num=300)
	Polynomial_DotsForSmoothLine = PolynomialFeatures(degree=4, include_bias=False).fit_transform(DotsForSmoothLine.reshape(300,1))
	y_predict = np.sum(reg.theta_parameters[1:]*Polynomial_DotsForSmoothLine, axis=1) + reg.theta_parameters[0]

	plt.plot(X, y, 'rX')
	plt.plot(DotsForSmoothLine, y_predict)

	plt.show()


# Prepare data
boston = load_boston()

x_RoomsPerDwelling = boston.data[:20,5:6]
Poly_x_RoomsPerDwelling = PolynomialFeatures(degree=4, include_bias=False).fit_transform(x_RoomsPerDwelling)
y_MedianPriceValue = boston.target[:20]


reg = MyRidgeRegression(alpha=1)
reg.fit(Poly_x_RoomsPerDwelling, y_MedianPriceValue)

DrawPlot(x_RoomsPerDwelling, y_MedianPriceValue)
