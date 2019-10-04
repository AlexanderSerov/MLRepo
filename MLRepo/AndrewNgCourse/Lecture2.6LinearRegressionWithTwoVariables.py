import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from sklearn.datasets import load_boston

#GradientDescent for y = m*x+b function. 
def GradientDescent(x, y):
	theta_zero = 0
	theta_one = 0
	learning_rate = 0.01
	theta_zero_temp = 0
	theta_one_temp = 0
	n = len(x)

	while (1):
		y_predicted = theta_one * x + theta_zero
		theta_one_derivative = (1/n)*sum(x*(y_predicted - y))
		theta_zero_derivative = (1/n)*sum(y_predicted - y)
		theta_one = theta_one - learning_rate * theta_one_derivative
		theta_zero = theta_zero - learning_rate * theta_zero_derivative
		print("theta_one =", theta_one, "| theta_zero = ", theta_zero)

		if (math.isclose(theta_one,theta_one_temp, rel_tol=1e-12) or math.isclose(theta_zero,theta_zero_temp, rel_tol=1e-12)):
			print("Break due to isclose function")
			break
		elif (abs(theta_one) > sys.maxsize):
			print("Break due to error function exceed")
			break

		theta_one_temp = theta_one
		theta_zero_temp = theta_zero
		
	print(theta_one, theta_zero)
	return theta_one, theta_zero

def DrawPlot(x,y):
	y_predict = Result[0]*x+Result[1]

	# #Axis scale.
	# min_x = np.amin(x) - 35
	# max_x = np.amax(x) + 35
	# min_y = np.amin(y) - 35
	# max_y = np.amax(y) + 35

	# plt.axis([min_x,max_x,min_y,max_y])
	plt.plot(x, y, 'rX')				#markersize='15'
	plt.plot(x, y_predict)

	plt.show()

# Price = np.array([100, 150, 110, 120, 200, 70])
# Foot = np.array([20, 40, 35, 40, 60, 18])

boston = load_boston()

x_RoomsPerDwelling = boston.data[:,5]
y_MedianPriceValue = boston.target

Result = GradientDescent(x_RoomsPerDwelling, y_MedianPriceValue)

DrawPlot(x_RoomsPerDwelling, y_MedianPriceValue)