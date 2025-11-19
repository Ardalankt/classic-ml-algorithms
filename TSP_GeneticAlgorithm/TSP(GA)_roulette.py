import math
import random
import numpy as np

citiesList = []
cityCoordinates = []
mutationRate = 0.1
crossoverRate = 0.9
populationSize = 100
numOfGenerations = 2000

tsp51 = open("TSP51.txt", "r")
strings = tsp51.read().split("\n")
for string in strings:
    if len(string) != 0:
        info = string.split(" ")
        citiesList.append(int(info[0]))
        cityCoordinates.append([int(info[1]), int(info[2])])

numOfCities = len(cityCoordinates)


def generateInitialPopulation(citiesList, size):
    initialPopulation = []
    for i in range(size):
        copyCities = citiesList.copy()
        random.shuffle(copyCities)
        initialPopulation.append(copyCities)

    return initialPopulation


def distanceBetweenTwoCities(city1, city2):
    distance = np.sqrt(
        ((cityCoordinates[city2 - 1][0] - cityCoordinates[city1 - 1][0]) ** 2)
        + ((cityCoordinates[city2 - 1][1] - cityCoordinates[city1 - 1][1]) ** 2)
    )
    return distance


def calculateTotalDistanceByChromosome(chromosome):
    totalDistance = 0
    for i in range(len(chromosome)):
        if i == len(chromosome) - 1:
            totalDistance += distanceBetweenTwoCities(chromosome[i], chromosome[0])
        else:
            totalDistance += distanceBetweenTwoCities(chromosome[i], chromosome[i + 1])

    return totalDistance


def calculateFitnessOfChromosome(population):
    totalDistanceOfChromosomes = []

    for i in range(len(population)):
        totalDistanceOfChromosomes.append(
            calculateTotalDistanceByChromosome(population[i])
        )

    maxDistance = max(totalDistanceOfChromosomes)
    chromosomeFitness = maxDistance - totalDistanceOfChromosomes
    chromosomeFitnessSum = sum(chromosomeFitness)
    chromosomesFitnessProbability = chromosomeFitness / chromosomeFitnessSum

    return chromosomesFitnessProbability


def rouletteWheelSelection(population, chromosomesFitnessProbability):
    chosenChromosome = random.choices(population, weights=chromosomesFitnessProbability)

    return chosenChromosome[0]


def singlePointCrossover(parentChromosome1, parentChromosome2):
    singleCrossoverPoint = random.randint(1, (numOfCities - 1))

    offspringChromosome1 = parentChromosome1[0:singleCrossoverPoint]

    for city in parentChromosome2:
        if city not in offspringChromosome1:
            offspringChromosome1.append(city)

    offspringChromosome2 = parentChromosome2[0:singleCrossoverPoint]
    for city in parentChromosome1:
        if city not in offspringChromosome2:
            offspringChromosome2.append(city)

    return offspringChromosome1, offspringChromosome2


def swapMutation(offspringChromosome):
    point1 = random.randint(0, numOfCities - 1)
    point2 = random.randint(0, numOfCities - 1)

    temporary = offspringChromosome[point1]
    offspringChromosome[point1] = offspringChromosome[point2]
    offspringChromosome[point2] = temporary
    return offspringChromosome


def geneticAlgorithm():
    population = generateInitialPopulation(citiesList, populationSize)
    chromosomesFitnessProbability = calculateFitnessOfChromosome(population)

    parents = []
    numOfChosenParents = int(crossoverRate * populationSize)
    for i in range(numOfChosenParents):
        parents.append(
            rouletteWheelSelection(population, chromosomesFitnessProbability)
        )

    offspringChromosomes = []
    for i in range(0, len(parents), 2):
        offspring1, offspring2 = singlePointCrossover(parents[i], parents[i + 1])

        if random.random() < mutationRate:
            offspring1 = swapMutation(offspring1)

        if random.random() < mutationRate:
            offspring2 = swapMutation(offspring2)

        offspringChromosomes.append(offspring1)
        offspringChromosomes.append(offspring2)

    elitismCheck = parents + offspringChromosomes
    elitismFitness = calculateFitnessOfChromosome(elitismCheck)
    fitnessIndex = np.argsort(elitismFitness)[::-1]
    elitismIndex = fitnessIndex[0:populationSize]
    elitismResult = []
    for index in elitismIndex:
        elitismResult.append(elitismCheck[index])

    if calculateTotalDistanceByChromosome(elitismCheck[fitnessIndex[0]]) < 500:
        return elitismResult

    for i in range(numOfGenerations):
        print(f"generation {i + 1} completed.")
        chromosomesFitnessProbability = calculateFitnessOfChromosome(elitismResult)

        parents = []
        numOfChosenParents = int(crossoverRate * populationSize)
        for i in range(numOfChosenParents):
            parents.append(
                rouletteWheelSelection(elitismResult, chromosomesFitnessProbability)
            )

        offspringChromosomes = []
        for i in range(0, len(parents), 2):
            offspring1, offspring2 = singlePointCrossover(parents[i], parents[i + 1])

            if random.random() < mutationRate:
                offspring1 = swapMutation(offspring1)

            if random.random() < mutationRate:
                offspring2 = swapMutation(offspring2)

            offspringChromosomes.append(offspring1)
            offspringChromosomes.append(offspring2)

        elitismCheck = parents + offspringChromosomes
        elitismFitness = calculateFitnessOfChromosome(elitismCheck)
        fitnessIndex = np.argsort(elitismFitness)[::-1]
        # We keep our best answers:
        elitismIndex = fitnessIndex[0 : int(0.7 * populationSize)]

        elitismResult = []
        for index in elitismIndex:
            elitismResult.append(elitismCheck[index])

        # We pick some random answers:
        notElitismIndices = []
        for j in range(int(0.3 * populationSize)):
            notElitismIndices.append(random.randint(0, populationSize - 1))
        for index in notElitismIndices:
            elitismResult.append(population[index])

        if calculateTotalDistanceByChromosome(elitismCheck[fitnessIndex[0]]) < 500:
            break

        random.shuffle(elitismResult)

    return elitismResult


result = geneticAlgorithm()
totalDistanceOfChromosomes = []

for i in range(populationSize):
    totalDistanceOfChromosomes.append(calculateTotalDistanceByChromosome(result[i]))

indicesOfPath = np.argmin(totalDistanceOfChromosomes)
shortestPath = min(totalDistanceOfChromosomes)
shortestPathIndices = result[indicesOfPath]
print("-----------------------------------------------------------")
print("Best answer in the final population is :" + "\n")
print("Best path found: ")
print(shortestPathIndices)
print("\n")
print(f"The cost of the path: {shortestPath}")
print("-----------------------------------------------------------")
