import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from numpy import linalg as LA

def NormalEquation(X, y):
	#For polynomial regression 1 will be added automatically.
	#if X.shape[1] == 1:
	# ones = np.ones((X.shape[0],1))
	# X = np.concatenate((ones,X), axis=1)

	return LA.matrix_power((X.T@X),-1)@X.T@y

def DrawPlot(x, y, x_AxisName, y_AxisName):
	y_predict = Result[0] + Result[1]*x

	plt.plot(x, y, 'rX')
	plt.plot(x, y_predict)
	plt.xlabel(x_AxisName)
	plt.ylabel(y_AxisName)

	plt.show()

def DrawPlotForQuadraticFunction(x, y, x_AxisName, y_AxisName):
	y_predict = Result[0] + Result[1]*x + Result[2]*(x**2)

	plt.plot(x, y, 'rX')
	plt.plot(x, y_predict, 'bo')
	plt.xlabel(x_AxisName)
	plt.ylabel(y_AxisName)

	plt.show()

# Price = np.array([100, 150, 110, 120, 200, 70])
# Foot = np.array([20, 40, 35, 40, 60, 18])

#Load and prepare data.
boston = load_boston()

#Preparations for quadratic function.
x_RoomsPerDwelling = np.array(boston.data[:,5:6])
y_MedianPriceValue = np.array([boston.target]).T
x_RoomsPerDwelling_Tranformed = PolynomialFeatures(degree=2).fit_transform(x_RoomsPerDwelling)
x_AxisName = boston.feature_names[5]
y_AxisName = 'Price'

# x_RoomsPerDwelling = np.asarray(boston.data[:,5:7])
# y_MedianPriceValue = np.asarray([boston.target]).T
# x_AxisName = boston.feature_names[5]
# y_AxisName = 'Price'

Result = NormalEquation(x_RoomsPerDwelling_Tranformed, y_MedianPriceValue)
print(Result)

DrawPlotForQuadraticFunction(x_RoomsPerDwelling, y_MedianPriceValue, x_AxisName, y_AxisName)