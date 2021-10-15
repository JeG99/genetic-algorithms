from deap import gp, base, creator, tools, algorithms
import matplotlib.pyplot as plt
import networkx as nx
import operator
import math
import random
import numpy as np
from numpy.core.numeric import Inf
from sklearn.metrics import mean_absolute_error

entradas = [(0, 10), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]
salidas = [90, 82, 74, 66, 58, 50, 42, 34, 26, 18]

def div(arg1, arg2):
    try:
        cociente = arg1 / arg2    
    except ZeroDivisionError:
        cociente = 1
    return cociente

pset = gp.PrimitiveSet("main", 2);
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(max, 2)
pset.addPrimitive(div, 2)

pset.addTerminal(math.pi, name="PI")
pset.addTerminal(math.e, name="E")
pset.addEphemeralConstant("RANDOM", lambda: random.randint(0, 10))
# pset.addEphemeralConstant(lambda: random.randint(-10, 10), int)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")

toolbox = base.Toolbox()

expr = gp.genFull(pset, min_=1, max_=5)

creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
creator.create('Individual', gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

#toolbox.register("expr", expr, pset=pset)
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=7)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

def eval_func(individual, input, output):
    function = gp.compile(individual, pset)
    pred = [function(i[0], i[1]) for i in input]
    return np.abs(mean_absolute_error(output, pred)),

toolbox.register('select', tools.selTournament, tournsize=2)
toolbox.register('mate', gp.cxOnePoint)
toolbox.register('mutate', gp.mutNodeReplacement, pset=pset)
toolbox.register('evaluate', eval_func, input=entradas, output=salidas)
toolbox.register('compile', gp.compile, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter('height'), max_value=17))

toolbox.register('population', tools.initRepeat, list, toolbox.individual)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register('min', np.min)
stats.register('max', np.max)
stats.register('mean', np.mean)
stats.register('std', np.std)

hof = tools.HallOfFame(1)
pop = toolbox.population(n=10)

results, log = algorithms.eaSimple(pop, toolbox, cxpb=1.0, mutpb=0.8, ngen=100, stats=stats, halloffame=hof, verbose=True)
print(hof)
#https://deap.readthedocs.io/en/master/api/tools.html#deap.gp.staticLimit

tree = nx.Graph()
nodes, edges, labels = gp.graph(hof[0])

