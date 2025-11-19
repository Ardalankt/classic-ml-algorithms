import random
import numpy as np


mutationRate = 0.1
numOfGenerations = 500
populationSize = 60
allCombinations = []


def initialPopulation():
    global allCombinations
    allPossibleSolutions = []
    for value in range(256):
        binaryNumber = bin(value)[2:]
        binaryNumber = ("0" * (8 - len(binaryNumber))) + binaryNumber
        allPossibleSolutions.append([binaryNumber, value])

    allCombinations = allPossibleSolutions
    initialPop = random.choices(allPossibleSolutions, k=populationSize)

    return initialPop


def calculateFitness(value):
    return value**2


# ---------------------------------------------------------------------------
# Roulette Wheel Selection


def calculateSumOfFitness(population):
    sum = 0
    for i in population:
        sum += calculateFitness(i[1])
    return sum


def calculateAverageFitness(population):
    sum = calculateSumOfFitness(population)
    return round(sum / populationSize)


def calculateActualCount(population):
    average = calculateAverageFitness(population)
    actualCount = []
    for i in range(len(population)):
        actualCount.append(round((calculateFitness(population[i][1]) / average)))
    return actualCount


# ---------------------------------------------------------------------------


def generateNewPopulation(population):
    actualCount = calculateActualCount(population)
    newPopulation = []
    for i in range(len(actualCount)):
        count = actualCount[i]
        while count != 0 and len(newPopulation) < 40:
            newPopulation.append(population[i])
            count -= 1
    while len(newPopulation) < populationSize:
        newPopulation.append(allCombinations[random.randint(0, 255)])
    return newPopulation


def singlePointCrossover(parent1, parent2):
    point = random.randint(1, 7)

    offspring1 = parent1[0][0:point] + parent2[0][point:]
    offspring2 = parent2[0][0:point] + parent1[0][point:]

    return offspring1, offspring2


def flipBitMutation(chromosome):
    point = random.randint(0, 7)
    result = ""

    if chromosome[point] == "0":
        for i in range(point):
            result += chromosome[i]

        result += "1"

        for i in range(point + 1, len(chromosome)):
            result += chromosome[i]

    else:
        for i in range(point):
            result += chromosome[i]

        result += "0"

        for i in range(point + 1, len(chromosome)):
            result += chromosome[i]

    return result


def geneticAlgorithm():
    initPop = initialPopulation()
    population = initPop
    for i in range((numOfGenerations)):
        before = population.copy()

        population = generateNewPopulation(before)

        newPopulation = []

        for j in range(0, populationSize, 2):
            offspring1, offspring2 = singlePointCrossover(
                population[j], population[j + 1]
            )

            check1 = random.random()
            if check1 < mutationRate:
                offspring1 = flipBitMutation(offspring1)

            check2 = random.random()
            if check2 < mutationRate:
                offspring2 = flipBitMutation(offspring2)

            newPopulation.append([offspring1, int(offspring1, 2)])
            newPopulation.append([offspring2, int(offspring2, 2)])

        population = newPopulation

    return population


result = geneticAlgorithm()
result = sorted(result, key=lambda x: x[1], reverse=True)
print(
    "---------------------------------------------------------------------------------------------------------------------"
)
print("The final population is : " + "\n")
for i in range(0, populationSize, 2):
    print(f"Chromosome {i+1}: {result[i]} , Chromosome {i+2}: {result[i+1]} ")

print("")
print(f"The best answer in the final population is: {result[0][1]} ")
print(f"f(x) = {result[0][1] ** 2}")
print(
    "---------------------------------------------------------------------------------------------------------------------"
)
