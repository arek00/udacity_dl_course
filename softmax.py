import numpy as np
import matplotlib.pyplot as plot

scores = [[[0.1, 0.2, 0.3, 6],
           [22, 14, 0.5, 1.6],
           [3, 8, 7, 6]],
          [[1, 2, 3, 6],
           [2, 4, 5, 6],
           [3, 8, 7, 6]],
          [[3, 1, 1, 1],
           [2, 4, 5, 6],
           [3, 8, 7, 6]]]

##My implementation of softmax

def softmax(matrix):
    expSum = 0
    for element in matrix:
        expSum += exponentBase(element)

    probabilities = []

    count = len(scores)
    for iterator in range(0, count):
        score = scores[iterator]
        probabilities.append(exponentBase(score) / expSum)

    return np.array(probabilities)


def exponentBase(number):
    return np.power(np.e, number)


print(softmax(scores))

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plot.plot(x, softmax(scores).T, linewidth=2)
plot.show()