from deap import base, creator
from deap import algorithms
from deap import tools
#from deap.tools.constraint import distance
import numpy as np
import pandas as pd
import random

#w = [23, 31, 29, 44, 53, 38, 63, 85, 89, 82]
#p = [92, 57, 49, 68, 60, 43, 67, 84, 87, 72]
w = pd.read_csv("p08_w.txt", header=None).rename(columns={0: 'weights'})
p = pd.read_csv("p08_p.txt", header=None).rename(columns={0: 'profits'})

MAX_WEIGHT = 6404180

def func_eval(individual):
    """
    En esta funcion se calcula la evaluacion y el peso. Despues se penaliza en caso de que se pase
    del peso permitido. En la tarea deben de hacer que solamente se calcule la evaluacion y que
    en otra se calcule el peso para calcular si la solucion es valida o no. Pueden usar
    tools.DeltaPenalty o la otra. Checar la lista de funciones en la libreria.

    :param u:
    :return:
    """
    return np.sum(np.asarray(individual) * np.asarray(p)),

def feasible(individual):
    if np.sum(np.asarray(individual) * np.asarray(w)) > MAX_WEIGHT:
        return False
    return True

def distance(individual):
    return np.abs(np.sum(np.asarray(individual) * np.asarray(p)) - 13549094) ** 2

# Definir si es un problema de maximizar o minimizar.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Definir que los individuos son listas y que se va a maximizar.
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Seleccionar la función de selección.
toolbox.register("select", tools.selRoulette)
# Seleccionar la función de mutación.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
# Seleccionar el de reproducción.
toolbox.register("mate", tools.cxOnePoint)
# Definir la función de evaluación.
toolbox.register("evaluate", func_eval)
# Definir la penalización.

toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 25916209, distance))
# toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 0))
# Definir un elemento del individuo.
toolbox.register("attribute", random.randint, a=0, b=1)
# Definiendo la creación de individuos como una lista de n elementos.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=24)
# Definiendo la creación de la población.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop = toolbox.population(n=100)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)

hof = tools.HallOfFame(3)

log = algorithms.eaSimple(population=pop, toolbox=toolbox, halloffame=hof, cxpb=0.4, mutpb=0.8,
                    ngen=24, stats=stats, verbose=True)

print(hof)
print([(func_eval(i)[0], np.sum(np.asarray(i) * np.asarray(w))) for i in hof])