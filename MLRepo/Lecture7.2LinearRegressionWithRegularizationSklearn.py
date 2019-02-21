import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


def DrawPlot(X,y):

	DotsForSmoothLine = np.linspace(np.amin(X), np.amax(X), num=300)
	Polynomial_DotsForSmoothLine = PolynomialFeatures(degree=5, include_bias=False).fit_transform(DotsForSmoothLine.reshape(300,1))
	y_predict = np.sum(reg.coef_*Polynomial_DotsForSmoothLine, axis=1) + reg.intercept_

	plt.plot(X, y, 'rX')
	plt.plot(DotsForSmoothLine, y_predict)

	plt.show()

# Price = np.array([[100, 150, 110, 120, 200, 70]]).T
# Poly_Price = PolynomialFeatures(degree=5, include_bias=False).fit_transform(Price)
# Foot = np.array([20, 40, 35, 40, 60, 18])

boston = load_boston()

x_RoomsPerDwelling = boston.data[:,5:6]
Poly_x_RoomsPerDwelling = PolynomialFeatures(degree=5, include_bias=False).fit_transform(x_RoomsPerDwelling)
y_MedianPriceValue = boston.target[:]

# reg = LinearRegression().fit(Poly_x_RoomsPerDwelling, y_MedianPriceValue)
reg = Ridge(alpha=100).fit(Poly_x_RoomsPerDwelling, y_MedianPriceValue)
print(reg.score(Poly_x_RoomsPerDwelling, y_MedianPriceValue))


DrawPlot(x_RoomsPerDwelling, y_MedianPriceValue)