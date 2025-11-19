import numpy as np
import math
import random

# Network structure
inputSize = 25
hiddenSize = 5
outputSize = 1

# Weights and biases
weightsHidden = np.random.uniform(-1, 1, size=(hiddenSize, inputSize))
weightsOutput = np.random.uniform(-1, 1, size=hiddenSize)
biasHidden = np.random.uniform(-1, 1, size=hiddenSize)
biasOutput = np.random.uniform(-1, 1, size=outputSize)


def stepFunction(input):
    if input >= 0:
        return +1
    else:
        return -1


def sigmoidFunction(x):
    # Output in range [-1, 1]
    return (2 / (1 + np.exp(-x))) - 1


def sigmoidFunctionDerivative(x):
    # Derivative compatible with sigmoidFunction above
    s = sigmoidFunction(x)
    return 0.5 * (1 + s) * (1 - s)


def getOutput(input):
    """
    Feed-forward for a single input.
    Returns the raw network output (continuous in [-1, 1]).
    """
    outputsHidden = np.array([0.0] * hiddenSize)

    # Hidden layer
    for j in range(hiddenSize):
        yNetInput = weightsHidden[j].dot(input) + biasHidden[j]
        outputsHidden[j] = sigmoidFunction(yNetInput)

    # Output layer (single neuron)
    yNetInputOutput = weightsOutput.dot(outputsHidden) + biasOutput[0]
    output = sigmoidFunction(yNetInputOutput)

    return np.array([output])


def predict(input):
    """
    Predict class label using step function on the final output.
    Returns -1 or +1.
    """
    output = getOutput(input)[0]
    return stepFunction(output)


def twoLayerPerceptron(trainingSet, learningRate, maxEpochs=1000, errorThreshold=0.001):
    """
    trainingSet[0] : inputs  (shape: [numSamples, 25])
    trainingSet[1] : targets (shape: [numSamples]) in range [-1, 1]
    """
    global weightsHidden, weightsOutput, biasHidden, biasOutput

    X = trainingSet[0]
    T = trainingSet[1]

    epoch = 0
    while epoch < maxEpochs:
        epoch += 1
        totalError = 0.0

        # Shuffle samples each epoch (optional but better)
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for idx in indices:
            x = X[idx]
            t = T[idx]

            # ----- Feed-forward -----
            outputsHidden = np.array([0.0] * hiddenSize)
            hiddenNetInput = np.array([0.0] * hiddenSize)

            # Hidden layer
            for j in range(hiddenSize):
                yNetInputHidden = weightsHidden[j].dot(x) + biasHidden[j]
                hiddenNetInput[j] = yNetInputHidden
                outputsHidden[j] = sigmoidFunction(yNetInputHidden)

            # Output layer (single neuron)
            yNetInputOutput = weightsOutput.dot(outputsHidden) + biasOutput[0]
            output = sigmoidFunction(yNetInputOutput)

            # Squared error for this sample
            totalError += 0.5 * (t - output) ** 2

            # ----- Backpropagation -----
            # Output delta
            deltaOutput = (t - output) * sigmoidFunctionDerivative(yNetInputOutput)

            # Hidden deltas
            deltasHidden = np.array([0.0] * hiddenSize)
            for j in range(hiddenSize):
                deltasHidden[j] = (
                    deltaOutput
                    * weightsOutput[j]
                    * sigmoidFunctionDerivative(hiddenNetInput[j])
                )

            # ----- Update weights and biases -----
            # Output weights and bias
            for j in range(hiddenSize):
                weightsOutput[j] = (
                    weightsOutput[j] + learningRate * deltaOutput * outputsHidden[j]
                )
            biasOutput[0] = biasOutput[0] + learningRate * deltaOutput

            # Hidden weights and biases
            for j in range(hiddenSize):
                for k in range(inputSize):
                    weightsHidden[j][k] = (
                        weightsHidden[j][k] + learningRate * deltasHidden[j] * x[k]
                    )
                biasHidden[j] = biasHidden[j] + learningRate * deltasHidden[j]

        print("Epoch:", epoch, " Total Error:", totalError)

        # Early stopping
        if totalError < errorThreshold:
            print("Training stopped early at epoch", epoch)
            break
