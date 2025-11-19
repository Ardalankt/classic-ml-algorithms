import numpy as np
import random

populationSize = 800
mutationRate = 0.1
numberOfGenerations = 600
MaxFitness = 28

# The positions of the queens are calculated from the bottom of the board .
# [3, 7, 0, 2, 5, 1, 6, 4] this means that the first queen is in the first column and 4(0 base index)
# squares up form bottom of the chess board --> [A4]


def generateRandomBoardInstance():
    boardInstance = []
    for column in range(8):
        row = random.randint(0, 7)
        boardInstance.append(row)
    return boardInstance


def generateInitialPopulation():
    initialPopulation = []
    for i in range(populationSize):
        initialPopulation.append(generateRandomBoardInstance())
    return initialPopulation


def calculateFitnessOfChromosome(boardInstance):
    horizontalCollisions = 0
    diagonalCollisions = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if boardInstance[i] == boardInstance[j]:
                horizontalCollisions += 1
            elif abs(boardInstance[i] - boardInstance[j]) == j - i:
                diagonalCollisions += 1
    fitnessOfChromosome = MaxFitness - (horizontalCollisions + diagonalCollisions)
    return fitnessOfChromosome


def singlePointCrossover(parent1, parent2):
    point = random.randint(1, 7)
    offspring1 = parent1[0:point] + parent2[point:]
    offspring2 = parent2[0:point] + parent1[point:]
    return offspring1, offspring2


def swapMutation(chromosome):
    point1 = random.randint(0, 7)
    point2 = random.randint(0, 7)
    temp = chromosome[point1]
    chromosome[point1] = chromosome[point2]
    chromosome[point2] = temp
    return chromosome


def tournamentSelection(population):
    tournamentSize = 5
    sample = random.sample(population, tournamentSize)
    sampleFitness = []
    for i in range(len(sample)):
        sampleFitness.append(calculateFitnessOfChromosome(sample[i]))

    fitnessIndices = np.argsort(sampleFitness)[::-1]
    return sample[fitnessIndices[0]]


def geneticAlgorithm():
    population = generateInitialPopulation()
    for i in range(numberOfGenerations):
        populationFitness = []
        for j in range(len(population)):
            populationFitness.append(calculateFitnessOfChromosome(population[j]))

        fitnessIndices = np.argsort(populationFitness)[::-1]
        if populationFitness[fitnessIndices[0]] == MaxFitness:
            break

        newPopulation = []

        # Elitism we pick 6 of the best solutions.
        newPopulation.append(population[fitnessIndices[0]])
        newPopulation.append(population[fitnessIndices[1]])
        newPopulation.append(population[fitnessIndices[2]])
        newPopulation.append(population[fitnessIndices[3]])
        newPopulation.append(population[fitnessIndices[4]])
        newPopulation.append(population[fitnessIndices[5]])

        while len(newPopulation) < populationSize:
            parent1 = tournamentSelection(population)
            parent2 = tournamentSelection(population)
            offspring1, offspring2 = singlePointCrossover(parent1, parent2)
            if random.random() < mutationRate:
                offspring1 = swapMutation(offspring1)

            if random.random() < mutationRate:
                offspring2 = swapMutation(offspring2)

            newPopulation.append(offspring1)
            newPopulation.append(offspring2)

        population = newPopulation

    return population


def drawChessBoardInstance(boardInstance):
    board = []
    columns = ["   A ", " B ", " C ", " D ", " E ", " F ", " G ", " H "]
    for _ in range(len(boardInstance)):
        board.append([" - "] * 8)

    for i in range(len(boardInstance)):
        board[(len(boardInstance) - 1) - boardInstance[i]][i] = " Q "

    rowNum = 8
    for r in board:
        print(str(rowNum) + " " + "".join(r))
        rowNum -= 1
    print("".join(columns))


population = geneticAlgorithm()
fitnessOfPopulation = []
for i in range(len(population)):
    fitnessOfPopulation.append(calculateFitnessOfChromosome(population[i]))

fitnessIndices = np.argsort(fitnessOfPopulation)[::-1]
print(
    "Best solution found in the final population :",
    population[fitnessIndices[0]],
)

bestFitness = fitnessOfPopulation[fitnessIndices[0]]

print("Fitness score of the best solution: ", bestFitness)
print(f"----  {MaxFitness - bestFitness}, Collisions occurred.  ----")

print("The board instance is :")
drawChessBoardInstance(population[fitnessIndices[0]])
