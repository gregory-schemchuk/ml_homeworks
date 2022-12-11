import numpy as np, random, operator, pandas as pd


# Classes

class Vertex:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, vertex):
        xDis = abs(self.x - vertex.x)
        yDis = abs(self.y - vertex.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Fit:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fit = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromVertex = self.route[i]
                toVertex = None
                if i + 1 < len(self.route):
                    toVertex = self.route[i + 1]
                else:
                    toVertex = self.route[0]
                pathDistance += fromVertex.distance(toVertex)
            self.distance = pathDistance
        return self.distance

    def routeFit(self):
        if self.fit == 0:
            self.fit = 1 / float(self.routeDistance())
        return self.fit


# Classes


# Functions


def initPopulation(popSize, vertexList):
    population = []
    for i in range(0, popSize):
        route = random.sample(vertexList, len(vertexList))
        population.append(route)
    return population


def rankRoutes(population):
    fitResults = {}
    for i in range(0, len(population)):
        fitResults[i] = Fit(population[i]).routeFit()
    return sorted(fitResults.items(), key=operator.itemgetter(1), reverse=True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fit"])
    df['cum_sum'] = df.Fit.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fit.sum()
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if random.random() < mutationRate:
            swapWith = int(random.random() * len(individual))

            vertex1 = individual[swapped]
            vertex2 = individual[swapWith]

            individual[swapped] = vertex2
            individual[swapWith] = vertex1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGen = mutatePopulation(children, mutationRate)
    return nextGen


def doGenAlg(initVertexes, popSize, eliteSize, mutationRate, generations):
    population = initPopulation(popSize, initVertexes)
    print("Vertexes:\n")
    for i in range(0, len(initVertexes)):
        print(repr(initVertexes[i]))

    for i in range(0, generations):
        population = nextGeneration(population, eliteSize, mutationRate)

    print("BEST DISTANCE: " + str(1 / rankRoutes(population)[0][1]))
    bestRouteIndex = rankRoutes(population)[0][0]
    bestRoute = population[bestRouteIndex]
    return bestRoute


# Functions


if __name__ == '__main__':
    vertexList = []

    for i in range(0, 25):
        vertexList.append(Vertex(x=int(random.random() * 180), y=int(random.random() * 180)))

    bestRoute = doGenAlg(vertexList, 100, 20, 0.01, 200)
    print("BEST ROUTE: ")
    for i in range(0, len(bestRoute)):
        print(bestRoute[i])

