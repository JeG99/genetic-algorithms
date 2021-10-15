from deap import base, creator
from deap import algorithms
from deap import tools
#from deap.tools.constraint import distance
import numpy as np
from numpy.core.defchararray import asarray
import pandas as pd
import random

w = pd.read_csv("p08_w.txt", header=None).rename(columns={0: 'weights'})
p = pd.read_csv("p08_p.txt", header=None).rename(columns={0: 'profits'})

MAX_WEIGHT = 6404180

def func_eval(individual):
    return np.sum(np.asarray(individual) * np.asarray(p['profits'].values.tolist())),

def feasible(individual):
    if np.sum(np.asarray(individual) * np.asarray(w['weights'].values.tolist())) > MAX_WEIGHT:
        return False
    return True

def distance(individual):
    return  np.sum(np.asarray(individual) * np.asarray(w['weights'].values.tolist())) - MAX_WEIGHT

# Definir si es un problema de maximizar o minimizar.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Definir que los individuos son listas y que se va a maximizar.
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Seleccionar la función de selección.
toolbox.register("select", tools.selRoulette)
#toolbox.register("select", tools.selTournament, tournsize=24)
# Seleccionar la función de mutación.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
# Seleccionar el de reproducción.
toolbox.register("mate", tools.cxOnePoint)
# Definir la función de evaluación.
toolbox.register("evaluate", func_eval)
# Definir la penalización.
toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 65731, distance))
# Definir un elemento del individuo.
toolbox.register("attribute", random.randint, a=0, b=1)
# Definiendo la creación de individuos como una lista de n elementos.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=24)
# Definiendo la creación de la población.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=24)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)

hof1 = tools.HallOfFame(5)
hof2 = tools.HallOfFame(5)
hof3 = tools.HallOfFame(5)

log1 = algorithms.eaSimple(population=pop, toolbox=toolbox, halloffame=hof1, cxpb=1.0, mutpb=0.8, ngen=10, stats=stats, verbose=True)
log2 = algorithms.eaMuPlusLambda(population=pop, toolbox=toolbox, mu=12, lambda_=24, cxpb=0.7, mutpb=0.3, ngen=10, stats=stats, halloffame=hof2, verbose=True)
log3 = algorithms.eaMuCommaLambda(population=pop, toolbox=toolbox, mu=12, lambda_=24, cxpb=0.7, mutpb=0.3, ngen=10, stats=stats, halloffame=hof3, verbose=True)

print('eaSimple:')
evals1 = [print("Individual: {}, Profit: {}, Weight: {}".format(i, func_eval(i)[0], np.sum(np.asarray(i) * np.asarray(w['weights'].values.tolist())))) for i in hof1]
print('eaMuPlusLambda:')
evals2 = [print("Individual: {}, Profit: {}, Weight: {}".format(i, func_eval(i)[0], np.sum(np.asarray(i) * np.asarray(w['weights'].values.tolist())))) for i in hof2]
print('eaMuCommaLambda:')
evals3 = [print("Individual: {}, Profit: {}, Weight: {}".format(i, func_eval(i)[0], np.sum(np.asarray(i) * np.asarray(w['weights'].values.tolist())))) for i in hof3]