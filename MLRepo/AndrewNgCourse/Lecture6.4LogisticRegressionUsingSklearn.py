import matplotlib.pyplot as plt
import numpy as np
from math import e
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


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

    plt.plot(DotsForSmoothSigmoid,Sigmoid((DotsForSmoothSigmoid*clf.coef_.reshape(1)+clf.intercept_)), 'g', linewidth=5)


def Sigmoid(x):

    return 1/(1+e**(-x))


# Input data for tumor size
# TumorSize = np.array([[10], [35], [45], [30], [20], [70], [89], [56], [87], [99]])
# Malignant = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

X, y = load_iris(return_X_y=True)
X = X[:100,0:1]            # Slice matrix
y = y[:100]

clf = LogisticRegression(solver='lbfgs').fit(X, y)

# print(clf.predict([[56]]))


DrawSigmoid(X)
DrawPlot(X, y)
plt.show()

