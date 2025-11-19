import numpy as np

weights = np.array([0.0] * 25)
bias = 0.0


def calculateYNetInput(input):
    yNetInput = input.dot(weights) + bias
    return yNetInput


def stepFunction(yNetInput):
    if yNetInput >= 0:
        return +1
    else:
        return -1


def hebb(trainingSet):
    global bias

    for i in range(len(trainingSet[0])):
        for j in range(len(trainingSet[0][i])):
            weights[j] = weights[j] + (trainingSet[0][i][j] * trainingSet[1][i])

        bias = bias + (1 * trainingSet[1][i])
