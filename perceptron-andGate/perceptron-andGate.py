import numpy as np
import matplotlib.pyplot as plt


inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
andOutputs = [1, -1, -1, -1]
weights = [0, 0]
bias = 0
learningRate = 1
threshold = 0.2


def calculateYNetInput(input):
    yNetInput = input.dot(weights) + bias
    return yNetInput


def stepFunction(yNetInput):
    if yNetInput > threshold:
        return 1
    elif yNetInput <= threshold and yNetInput >= -threshold:
        return 0
    else:
        return -1


def perceptron():
    global bias
    error = True
    while error != False:
        error = False
        for i in range(len(inputs)):
            yNetInput = calculateYNetInput(inputs[i])

            print(
                f"X1: {str(inputs[i][0])} , X2: {str(inputs[i][1])} , Y-Net-Input:{yNetInput} , Y-Output:{stepFunction(yNetInput)} , training-output:{andOutputs[i]}"
            )
            print(
                f"Weight1 before updating: {weights[0]} , Weight2 before updating: {weights[1]} , Bias before updating: {bias} "
            )

            if stepFunction(yNetInput) != andOutputs[i]:
                error = True

                weights[0] = weights[0] + (learningRate * andOutputs[i] * inputs[i][0])
                weights[1] = weights[1] + (learningRate * andOutputs[i] * inputs[i][1])
                bias = bias + 1 * learningRate * andOutputs[i]

                print("Weight1 updated: " + str(weights[0]))
                print("Weight2 updated: " + str(weights[1]))
                print("Bias updated: " + str(bias))
                print("----------------------------------------")
            else:
                print("Weights and Bias not updated")
                print("----------------------------------------")

            x1 = np.linspace(-2, 2)
            line1 = (
                ((-weights[0] * x1) / weights[1])
                + (-bias / weights[1])
                + threshold / weights[1]
            )
            line2 = (
                ((-weights[0] * x1) / weights[1])
                + (-bias / weights[1])
                + -(threshold / weights[1])
            )
            plt.plot(x1, line1)
            plt.plot(x1, line2)
            plt.plot(inputs[0][0], inputs[0][1], "+")
            plt.plot(inputs[1:, 0], inputs[1:, 1], "_")
            if weights[1] == 0:
                plt.title("Divide by zero occurred!")
            else:
                plt.title("Perceptron")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.show()


perceptron()
print("Test inputs: ")
print(f"Inputs:{inputs[1]} Answer: {stepFunction(calculateYNetInput(inputs[1]))}")
