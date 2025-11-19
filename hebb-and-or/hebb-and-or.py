import matplotlib.pyplot as plt
import numpy as np

# Inputs for AND / OR (bipolar form)
inputs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])

andOutputs = [1, -1, -1, -1]
orOutputs = [1, 1, 1, -1]

# Weights and bias
weights = np.array([0.0, 0.0])
bias = 0.0


def calculateYNetInput(input):
    return input.dot(weights) + bias


def stepFunction(yNetInput):
    return 1 if yNetInput >= 0 else -1


def hebb(outputs):
    global weights, bias

    # Reset (optional)
    weights = np.array([0.0, 0.0])
    bias = 0.0

    # Train
    for i in range(len(inputs)):
        print("----------------------------------------")
        print(f"X1: {inputs[i][0]} , X2: {inputs[i][1]} , Y: {outputs[i]}")
        print(f"Before update -> w1: {weights[0]}, w2: {weights[1]}, bias: {bias}")

        # Hebb rule
        weights[0] += inputs[i][0] * outputs[i]
        weights[1] += inputs[i][1] * outputs[i]
        bias += outputs[i]

        print(f"After update  -> w1: {weights[0]}, w2: {weights[1]}, bias: {bias}")
        print("----------------------------------------")

        # Plot decision boundary after each update
        plt.figure()

        if weights[1] != 0:
            x1 = np.linspace(-2, 2)
            x2 = (-weights[0] * x1 - bias) / weights[1]
            plt.plot(x1, x2, label="Decision Boundary")
        else:
            plt.title("Divide by Zero (Cannot Draw Line)")

        # Plot points
        for j in range(len(inputs)):
            marker = "g+" if outputs[j] == 1 else "rx"
            plt.plot(inputs[j][0], inputs[j][1], marker)

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("Hebbian Learning")
        plt.grid(True)
        plt.show()


# Ask which gate to train
choice = input("Choose logic gate (AND / OR): ").strip().lower()

while choice not in ["and", "or"]:
    choice = input("Please type AND or OR: ").strip().lower()

outputs = andOutputs if choice == "and" else orOutputs

# Train
hebb(outputs)

# Testing
print("Test inputs:")
print("input: [0.9, -0.9] →", stepFunction(calculateYNetInput(np.array([0.9, -0.9]))))
print("input: [0.7, -0.7] →", stepFunction(calculateYNetInput(np.array([0.7, -0.7]))))
