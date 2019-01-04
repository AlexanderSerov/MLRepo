import matplotlib.pyplot as plt
import numpy as np
import math

# GradientDescent for y = m*x function.


def GradientDescent(x, y):
    theta_one = 0
    learning_rate = 0.0001
    MAX_IRETATION = 100000
    iteration_counter = 0
    theta_temp = 0
    n = len(x)

    while 1:
        y_predicted = theta_one * x
        # Init: -(2/n)*sum(x*(y - y_predicted))
        theta_one_derivative = (1 / n) * sum(x * (y_predicted - y))
        theta_one = theta_one - learning_rate * theta_one_derivative
        print("theta_one =", theta_one)

        if (math.isclose(theta_one, theta_temp, rel_tol=1e-20)):
            print("Break due to isclose function")
            break
        elif (iteration_counter >= MAX_IRETATION):
            print("Break due to iterator exceed max")
            break

        theta_temp = theta_one
        iteration_counter += 1

    return theta_one


def DrawPlot(x, y):
    max_x = np.amax(x) * 1.5
    min_x = np.amin(x) * 0.5

    plt.plot(x, y, 'rX', markersize='15')
    plt.plot([min_x, max_x], [min_x * Result, max_x * Result])
    plt.show()

# Prediction for y = m*x


def Predict(x):
    prediction = x * Result
    return prediction


# Input data for Real task
Price = np.array([100, 150, 110, 120, 200, 70])
Foot = np.array([20, 40, 35, 40, 60, 18])
# Input data from lesson
MathScore = np.array([92, 56, 88, 70, 80, 49, 65, 35, 66, 67])
CSScore = np.array([98, 68, 81, 80, 83, 52, 66, 30, 68, 73])


Result = GradientDescent(MathScore, CSScore)

print("Prediction = ", Predict(40))

DrawPlot(MathScore, CSScore)
