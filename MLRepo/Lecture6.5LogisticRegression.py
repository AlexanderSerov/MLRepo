import matplotlib.pyplot as plt
import numpy as np
from math import e
import math
import sys
from sklearn.datasets import load_iris
from sklearn import preprocessing
import copy


class MyLogisticRegression:

    theta_parameters = None

    def __init__(self, OneVsRest=False):

        self.OneVsRest = OneVsRest

    def fit(self, X, y):

        # If y has more than 2 labels
        if self.OneVsRest:
            for number in set(y):
                temp_y = copy.copy(y)
                for i, num in enumerate(temp_y):
                    if num == number:
                        temp_y[i] = 1
                    else:
                        temp_y[i] = 0
                self.GradientDescend(X, temp_y)

        else:
            self.GradientDescend(X, y)

    def predict(self, x):
        
        Result = []

        ones = np.ones((x.shape[0],1))
        x = np.concatenate((ones,x), axis=1)

        # Make prediction for each element in x matrix
        for row in x:
            # If OneVsRest setted then we need compute sigmoid for each hypothesis and select greater
            if self.OneVsRest:
                Result.append(np.argmax(self.Sigmoid(np.sum(row*self.theta_parameters, axis=1))))
            else:
                # If there is one hypothesis then make prediction
                if np.sum(row*[self.theta_parameters], axis=1) > 0:
                    Result.append(1)
                elif np.sum(row*[self.theta_parameters], axis=1) <= 0:
                    Result.append(0)

        return Result

    def GradientDescend(self, X, y):

        # Feature scaling
        # X = self.MeanNormalization(X)
        # X = preprocessing.minmax_scale(X, feature_range=(-0.5, 0.5))

        ones = np.ones((X.shape[0],1))
        X = np.concatenate((ones,X), axis=1)

        theta_vector = np.zeros(X.shape[1])     #Fill vector with zero values
        learning_rate = 0.3
        theta_temp = 0

        while (1):
            theta_derivatives = (1/len(X))*np.sum(X.T * (self.Sigmoid(X @ theta_vector) - y), axis=1)
            theta_vector = theta_vector - learning_rate * theta_derivatives

            for idx, item in enumerate(theta_vector):
                print(" theta_", idx, " = ", item, sep='', end='')
            print('')

            if (math.isclose(theta_vector[1],theta_temp, rel_tol=1e-8)):
                print("Break due to isclose function")
                break
            elif (abs(theta_vector[1]) > sys.maxsize):
                print("Break due to error function exceed")
                break

            theta_temp = theta_vector[1]
                
        print(theta_vector)

        # Concatenate arrays if multy-label regression used
        if self.theta_parameters is None:
            self.theta_parameters = theta_vector
        else:
            self.theta_parameters = np.vstack((self.theta_parameters,theta_vector))


    @staticmethod
    def Sigmoid(x):

        return 1/(1+e**(-x))

    def MeanNormalization(self, X):

        Range = np.max(X) - np.min(X)      #Range in data set

        return (X - np.min(X)) / Range


def DrawPlot(X,y):

    plt.plot(X, y, 'rX', markersize='15')
    plt.xlabel("TumorSize")
    plt.ylabel("Malignant")
    plt.yticks((0,0.5,1))


def DrawSigmoid(X):

    TumorSize_max = np.amax(X)
    TumorSize_min = np.amin(X)
    DotsForSmoothSigmoid = np.linspace(TumorSize_min, TumorSize_max, num=300)

    # Draw middle separate line
    plt.plot((TumorSize_min,TumorSize_max),(0.5,0.5), 'k--')

    plt.plot(DotsForSmoothSigmoid,MyLogisticRegression.Sigmoid((DotsForSmoothSigmoid*clf.theta_parameters[1]+clf.theta_parameters[0])), 'g', linewidth=5)
    # plt.plot(DotsForSmoothSigmoid, Sigmoid(DotsForSmoothSigmoid*(-6.5507792320)+20.22649802), 'g', linewidth=5)



# Prepare data from iris dataset
X, y = load_iris(return_X_y=True)
X = X[:,1:2]            # Slice matrix
# y = y[:100]

# Train model
clf = MyLogisticRegression(OneVsRest=True)
clf.fit(X, y)
print(clf.theta_parameters)

# Test model
X_test = X[36:39]
print(clf.predict(X_test))


# DrawSigmoid(X)
# DrawPlot(X, y)
# plt.show()

