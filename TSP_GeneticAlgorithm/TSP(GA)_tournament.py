import math
import random
import numpy as np

citiesList = []
cityCoordinates = []
mutationRate = 0.05
crossoverRate = 0.9
populationSize = 300
numOfGenerations = 1000

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


def calculateFitnessOfChromosome(chromosome):
    distance = calculateTotalDistanceByChromosome(chromosome)
    fitness = 1 / distance
    return fitness


def tournamentSelection(population):
    tournamentSize = 6
    sample = random.sample(population, tournamentSize)
    sampleFitness = []
    for i in range(len(sample)):
        sampleFitness.append(calculateFitnessOfChromosome(sample[i]))

    fitnessIndices = np.argsort(sampleFitness)[::-1]
    return sample[fitnessIndices[0]]


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
    initialPopulation = generateInitialPopulation(citiesList, populationSize)
    population = initialPopulation

    for generation in range(numOfGenerations):
        print(f"generation {generation+1} completed.")
        fitnessOfChromosomes = []
        for i in range(len(population)):
            fitnessOfChromosomes.append(calculateFitnessOfChromosome(population[i]))

        fitnessIndices = np.argsort(fitnessOfChromosomes)[::-1]
        if calculateTotalDistanceByChromosome(population[fitnessIndices[0]]) < 500:
            return population

        bestAnswers = []
        # Elitism: we pick 10 of the best answers
        for i in range(10):
            bestAnswers.append(population[fitnessIndices[i]])

        parents = []
        numOfChosenParents = int(crossoverRate * populationSize)
        for i in range(numOfChosenParents):
            parents.append(tournamentSelection(population))

        offspringChromosomes = []
        for i in range(0, len(parents), 2):
            offspring1, offspring2 = singlePointCrossover(parents[i], parents[i + 1])

            if random.random() < mutationRate:
                offspring1 = swapMutation(offspring1)

            if random.random() < mutationRate:
                offspring2 = swapMutation(offspring2)

            offspringChromosomes.append(offspring1)
            offspringChromosomes.append(offspring2)

        population = offspringChromosomes
        # Elitism: we pick 10 of the best answers
        for i in range(len(bestAnswers)):
            population.append(bestAnswers[i])

        while len(population) < populationSize:
            population.append(initialPopulation[random.randint(0, populationSize - 1)])

    return population


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
