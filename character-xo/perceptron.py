import numpy as np

weights = np.array([0.0] * 25)
bias = 0.0
perceptronLearningRate = None
perceptronThreshold = None
perceptronMaxEpochs = 1000


def calculateYNetInput(input):
    yNetInput = input.dot(weights) + bias
    return yNetInput


def stepFunction(yNetInput):
    if yNetInput > perceptronThreshold:
        return 1
    elif yNetInput <= perceptronThreshold and yNetInput >= -perceptronThreshold:
        return 0
    else:
        return -1


def perceptron(trainingSet, learningRate, threshold, maxEpochs=1000):
    global perceptronLearningRate, perceptronThreshold, perceptronMaxEpochs, bias
    perceptronLearningRate = learningRate
    perceptronThreshold = threshold
    perceptronMaxEpochs = maxEpochs

    error = True
    epoch = 0

    while error != False and epoch < perceptronMaxEpochs:
        error = False
        epoch += 1

        for i in range(len(trainingSet[0])):
            x = trainingSet[0][i]
            target = trainingSet[1][i]

            yNetInput = calculateYNetInput(x)
            yOutput = stepFunction(yNetInput)

            if yOutput != target:
                error = True
                delta = perceptronLearningRate * (target - yOutput)

                for j in range(len(x)):
                    weights[j] = weights[j] + delta * x[j]

                bias = bias + delta * 1
