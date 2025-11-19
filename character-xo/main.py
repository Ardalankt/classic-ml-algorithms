import tkinter as tk
import numpy as np
import hebb as hebb
import perceptron as perceptron
import multiCategoryPerceptron as multiCategoryPerceptron
import adaline as adaline
import twoLayerPerceptron as twoLayerPerceptron

checkboxes = []
inputs = []
outputs = []
tkCharacters = ["X", "O"]


trainingDataRead = open("trainingData.txt", "r")
strings = trainingDataRead.read().split("\n")
for i in range(len(strings)):
    if len(strings[i]) != 0:
        inputs.append([])
        characters = strings[i].split(" ")
        for j in range(len(characters) - 1):
            if characters[j] == "+1":
                inputs[i].append(1)
            elif characters[j] == "-1":
                inputs[i].append(-1)

        if characters[len(characters) - 1] == "+1":
            outputs.append(1)
        elif characters[len(characters) - 1] == "-1":
            outputs.append(-1)
trainingDataRead.close()

trainingInputs = np.array(inputs)
trainingOutputs = np.array(outputs)
trainingSet = [trainingInputs, trainingOutputs]


def runAlgorithms():
    hebb.hebb(trainingSet)

    perceptron.perceptron(trainingSet, learningRate=1, threshold=0.2)

    multiCategoryPerceptron.multiCategoryPerceptron(
        trainingSet, learningRate=0.1, threshold=0.2
    )

    adaline.adaline(trainingSet, learningRate=0.1)

    twoLayerPerceptron.twoLayerPerceptron(trainingSet, 0.4)


# run algorithms
runAlgorithms()


def train():
    global trainingSet
    inputs = []
    outputs = []

    trainingDataAppend = open("trainingData.txt", "a")
    for checkbox in checkboxes:
        if checkbox[1].get() == 1:
            trainingDataAppend.write("+1 ")
        else:
            trainingDataAppend.write("-1 ")

    if varOption.get().lower() == "x":
        trainingDataAppend.write("+1")

    else:
        trainingDataAppend.write("-1")

    trainingDataAppend.write("\n")
    trainingDataAppend.close()

    trainingDataRead = open("trainingData.txt", "r")
    strings = trainingDataRead.read().split("\n")
    for i in range(len(strings)):
        if len(strings[i]) != 0:
            inputs.append([])
            characters = strings[i].split(" ")
            for j in range(len(characters) - 1):
                if characters[j] == "+1":
                    inputs[i].append(1)
                elif characters[j] == "-1":
                    inputs[i].append(-1)

            if characters[len(characters) - 1] == "+1":
                outputs.append(1)
            elif characters[len(characters) - 1] == "-1":
                outputs.append(-1)
    trainingDataRead.close()

    trainingInputs = np.array(inputs)
    trainingOutputs = np.array(outputs)
    trainingSet = [trainingInputs, trainingOutputs]

    # run algorithms
    runAlgorithms()


def predict():
    input = []
    for checkbox in checkboxes:
        if checkbox[1].get() == 1:
            input.append(1)
        else:
            input.append(-1)
    if varOption2.get() == "Hebb":
        # hebb
        yNetInputHebb = hebb.calculateYNetInput(np.array(input))
        resultHebb = hebb.stepFunction(yNetInputHebb)
        indexHebb = None
        if resultHebb == 1:
            indexHebb = 0
        else:
            indexHebb = 1

        labelOutput.config(text=f"Prediction: {tkCharacters[indexHebb]}")

    elif varOption2.get() == "Perceptron":
        # perceptron
        yNetInputPerceptron = perceptron.calculateYNetInput(np.array(input))
        resultPerceptron = perceptron.stepFunction(yNetInputPerceptron)
        indexPerceptron = None

        if resultPerceptron == 1:
            indexPerceptron = 0
            labelOutput.config(text=f"Prediction: {tkCharacters[indexPerceptron]}")
        elif resultPerceptron == -1:
            indexPerceptron = 1
            labelOutput.config(text=f"Prediction: {tkCharacters[indexPerceptron]}")
        else:
            labelOutput.config(text="Prediction: Neither")

    elif varOption2.get() == "MultiCategoryPerceptron":
        # multiCategoryPerceptron
        yNetInputX = multiCategoryPerceptron.calculateYNetInputX(np.array(input))
        yNetInputO = multiCategoryPerceptron.calculateYNetInputO(np.array(input))
        resultX = multiCategoryPerceptron.stepFunction(yNetInputX)
        resultO = multiCategoryPerceptron.stepFunction(yNetInputO)
        indexMultiCategory = None
        # print(resultX, resultO)
        if resultX == 1 and resultO == -1:
            indexMultiCategory = 0
            labelOutput.config(text=f"Prediction: {tkCharacters[indexMultiCategory]}")
        elif resultO == 1 and resultX == -1:
            indexMultiCategory = -1
            labelOutput.config(text=f"Prediction: {tkCharacters[indexMultiCategory]}")
        else:
            labelOutput.config(text="Prediction: Neither")

    elif varOption2.get() == "Adaline":
        # adaline
        yNetInputAdaline = adaline.calculateYNetInput(np.array(input))
        resultAdaline = adaline.stepFunction(yNetInputAdaline)
        indexAdaline = None
        if resultAdaline == 1:
            indexAdaline = 0
        else:
            indexAdaline = 1

        labelOutput.config(text=f"Prediction: {tkCharacters[indexAdaline]}")

    elif varOption2.get() == "TwoLayerPerceptron":
        # TwoLayerPerceptron
        outputTwoLayer = twoLayerPerceptron.getOutput(np.array(input))
        resultTwoLayerPerceptron = twoLayerPerceptron.stepFunction(outputTwoLayer)
        indexTwoLayer = None
        if resultTwoLayerPerceptron == 1:
            indexTwoLayer = 0
        else:
            indexTwoLayer = 1

        labelOutput.config(text=f"Prediction: {tkCharacters[indexTwoLayer]}")

    else:
        labelOutput.config(text="Choose" + "\n" + "Method")


# UI
root = tk.Tk()
root.geometry("600x200")
root.resizable(False, False)
root.title("Character Recognition")

for i in range(25):
    var = tk.IntVar()
    checkboxes.append(
        [
            tk.Checkbutton(
                root,
                variable=var,
                onvalue=1,
                offvalue=-1,
            ),
            var,
        ]
    )
    checkboxes[i][0].pack()
    checkboxes[i][0].place(x=((i % 5) * 40) + 10, y=(int((i / 5)) * 40) + 10)


button1 = tk.Button(root, text="train", height=1, width=3, command=train)
button1.pack()
button1.place(x=455, y=90)

button2 = tk.Button(root, text="predict", height=1, width=3, command=predict)
button2.pack()
button2.place(x=230, y=68)

varOption = tk.StringVar()
varOption.set("X")
optionMenu = tk.OptionMenu(
    root,
    varOption,
    *tkCharacters,
)
optionMenu.config(fg="white", font=("arial", "16"), borderwidth=0)
optionMenu.pack()
optionMenu.place(x=458, y=62)

tkAlgorithms = [
    "Hebb",
    "Perceptron",
    "MultiCategoryPerceptron",
    "Adaline",
    "TwoLayerPerceptron",
]
varOption2 = tk.StringVar()
varOption2.set("Method")
optionMenu2 = tk.OptionMenu(root, varOption2, *tkAlgorithms)
optionMenu2.pack()
optionMenu2.place(x=230, y=40)
optionMenu2.config(
    fg="white",
    font=("arial", "14"),
)

labelOutput = tk.Label(root, text="Prediction: ")
labelOutput.pack()
labelOutput.place(x=230, y=100)

labelOutputDescription = tk.Label(root, text="Prediction Section: ")
labelOutputDescription.pack()
labelOutputDescription.place(x=230, y=10)

labelTrainingDescription = tk.Label(root, text="Training Section: ")
labelTrainingDescription.pack()
labelTrainingDescription.place(x=455, y=10)

labelChoose = tk.Label(root, text="Choose Label")
labelChoose.pack()
labelChoose.place(x=455, y=35)

root.mainloop()
