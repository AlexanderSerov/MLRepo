import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from sklearn.datasets import load_boston
from sklearn.preprocessing import normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

#GradientDescent for y = m*X+b function.
def GradientDescent(X, y):
	X = normalize(X)			#Normalize raw data

	# ones = np.ones((X.shape[0],1))
	# X = np.concatenate((ones,X), axis=1)

	theta_vector = np.zeros((X.shape[1],1))		#Fill vector with zero values
	learning_rate = 0.1
	theta_temp = 0

	while (1):
		theta_derivatives = np.reshape((1/len(X))*np.sum(X * (X @ theta_vector - y), axis=0),(X.shape[1],1))
		theta_vector = theta_vector - learning_rate * theta_derivatives
		for idx, item in enumerate(theta_vector):
			print(" theta_", idx, " = ", item, sep='', end='')
		print('')

		if (math.isclose(theta_vector[0],theta_temp, rel_tol=1e-12)):
			print("Break due to isclose function")
			break
		elif (abs(theta_vector[0]) > sys.maxsize):
			print("Break due to error function exceed")
			break

		theta_temp = theta_vector[0]
				
	print(theta_vector)
	return theta_vector

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

def Draw3DPlot(x_RoomsPerDwelling, z_proportionOfOwnerOccupiedUnitsBuilt, y_MedianPriceValue):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	ax.scatter(x_RoomsPerDwelling, y_MedianPriceValue, z_proportionOfOwnerOccupiedUnitsBuilt,c='blue', marker='o', alpha=0.5)
	ax.plot_surface(x_RoomsPerDwelling,y_MedianPriceValue,z_proportionOfOwnerOccupiedUnitsBuilt, color='None', alpha=0.01)
	ax.set_xlabel('rooms per dwelling')
	ax.set_ylabel('Price')
	ax.set_zlabel('proportion of owner-occupied units built')

	plt.show()

# Price = np.array([100, 150, 110, 120, 200, 70])
# Foot = np.array([20, 40, 35, 40, 60, 18])



# Price = np.array([100, 150, 110, 120, 200, 70])
# Foot = np.array([20, 40, 35, 40, 60, 18])

#Load and prepare data.
boston = load_boston()

#Preparations for quadratic function.
x_RoomsPerDwelling = np.asarray(boston.data[:,5:6])
y_MedianPriceValue = np.asarray([boston.target]).T
x_RoomsPerDwelling_Tranformed = PolynomialFeatures(degree=2).fit_transform(x_RoomsPerDwelling)
x_AxisName = boston.feature_names[5]
y_AxisName = 'Price'


# x_RoomsPerDwelling = np.asarray(boston.data[:,5])
# z_proportionOfOwnerOccupiedUnitsBuilt = np.asarray(boston.data[:,6])
# y_MedianPriceValue = np.asarray([boston.target]).T
# x_AxisName = boston.feature_names[5]
# y_AxisName = 'Price'

#Result = GradientDescent(x_RoomsPerDwelling_Tranformed, y_MedianPriceValue)
Result = ([74.00, -321.60, 73.20])

DrawPlotForQuadraticFunction(x_RoomsPerDwelling, y_MedianPriceValue, x_AxisName, y_AxisName)