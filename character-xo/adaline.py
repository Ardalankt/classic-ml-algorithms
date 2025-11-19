import numpy as np
import math

weights = np.array([0.0] * 25)
bias = 0.0
adalineLearningRate = None


def calculateYNetInput(input):
    yNetInput = input.dot(weights) + bias
    return yNetInput


def stepFunction(yNetInput):
    if yNetInput >= 0:
        return +1
    else:
        return -1


def adaline(trainingSet, learningRate):
    global adalineLearningRate, bias
    adalineLearningRate = learningRate

    error = 60
    maxEpochs = 1000
    epoch = 0
    while error > 0.5 and epoch < maxEpochs:
        epoch += 1

        totalError = 0
        for i in range(len(trainingSet[0])):
            yNetInput = calculateYNetInput(trainingSet[0][i])
            totalError += (trainingSet[1][i] - yNetInput) ** 2

            for j in range(len(trainingSet[0][i])):
                deltaWeights = (
                    (adalineLearningRate / len(trainingSet[0]))
                    * (trainingSet[1][i] - yNetInput)
                    * trainingSet[0][i][j]
                )
                weights[j] = weights[j] + deltaWeights

            bias = bias + (
                (adalineLearningRate / len(trainingSet[0]))
                * (trainingSet[1][i] - yNetInput)
                * 1
            )
        error = totalError
