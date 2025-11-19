import numpy as np

# There are two categories : X , O
weights = np.array([[0.0] * 25, [0.0] * 25])
# bias[0] for category X and bias[1] for category O
bias = np.array([0.0, 0.0])
perceptronLearningRate = None
perceptronThreshold = None
perceptronMaxEpochs = 1000


def calculateYNetInputX(input):
    yNetInput = input.dot(weights[0]) + bias[0]
    return yNetInput


def calculateYNetInputO(input):
    yNetInput = input.dot(weights[1]) + bias[1]
    return yNetInput


def stepFunction(yNetInput):
    if yNetInput > perceptronThreshold:
        return 1
    elif yNetInput <= perceptronThreshold and yNetInput >= -perceptronThreshold:
        return 0
    else:
        return -1


def multiCategoryPerceptron(trainingSet, learningRate, threshold, maxEpochs=1000):
    global perceptronLearningRate, perceptronThreshold, perceptronMaxEpochs, bias
    perceptronLearningRate = learningRate
    perceptronThreshold = threshold
    perceptronMaxEpochs = maxEpochs

    # trainingSet[0] : inputs
    # trainingSet[1] : targets  (+1 for X, -1 for O)

    error = True
    epoch = 0

    while error != False and epoch < perceptronMaxEpochs:
        error = False
        epoch += 1

        for i in range(len(trainingSet[0])):
            x = trainingSet[0][i]
            target = trainingSet[1][i]  # +1 → X, -1 → O

            # X neuron
            yNetInputX = calculateYNetInputX(x)
            xOutput = stepFunction(yNetInputX)
            targetX = target  # +1 if X, -1 if O

            if xOutput != targetX:
                error = True
                deltaX = perceptronLearningRate * (targetX - xOutput)
                for j in range(len(x)):
                    weights[0][j] = weights[0][j] + deltaX * x[j]
                bias[0] = bias[0] + deltaX * 1

            # O neuron
            yNetInputO = calculateYNetInputO(x)
            oOutput = stepFunction(yNetInputO)
            targetO = -target  # +1 if O, -1 if X

            if oOutput != targetO:
                error = True
                deltaO = perceptronLearningRate * (targetO - oOutput)
                for j in range(len(x)):
                    weights[1][j] = weights[1][j] + deltaO * x[j]
                bias[1] = bias[1] + deltaO * 1
