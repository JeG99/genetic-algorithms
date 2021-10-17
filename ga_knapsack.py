from deap import base, creator
from deap import algorithms
from deap import tools
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import asarray
import pandas as pd
import random

def greater(row):
    df.at[row.name, 'greater'] = max(row['max'], df.iloc[row.name - 1].greater)
    return None

w = pd.read_csv("p08_w.txt", header=None).rename(columns={0: 'weights'})
p = pd.read_csv("p08_p.txt", header=None).rename(columns={0: 'profits'})

eaSimple = pd.DataFrame()
eaMuPlusLambda = pd.DataFrame()
eaMuCommaLambda = pd.DataFrame()

EXPERIMENTS = 10
NGEN = 1000
MAX_WEIGHT = 6404180

def func_eval(individual):
    return np.sum(np.asarray(individual) * np.asarray(p['profits'].values.tolist())),

def feasible(individual):
    if np.sum(np.asarray(individual) * np.asarray(w['weights'].values.tolist())) > MAX_WEIGHT:
        return False
    return True

def distance(individual):
    return abs(np.sum(np.asarray(individual) * np.asarray(w['weights'].values.tolist())) - MAX_WEIGHT)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("evaluate", func_eval)
toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 65731, distance))
toolbox.register("attribute", random.randint, a=0, b=1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=24)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("max", np.max)
stats.register("min", np.min)
stats.register("avg", np.mean)
stats.register("std", np.std)

hof1 = tools.HallOfFame(3)
hof2 = tools.HallOfFame(3)
hof3 = tools.HallOfFame(3)

def simple(pop):
    SIMPLE_MUTPB = 0.1
    SIMPLE_CXPB = 0.5
    
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
    toolbox.register("mate", tools.cxOnePoint)
    return algorithms.eaSimple(population=pop, toolbox=toolbox, halloffame=hof1, cxpb=SIMPLE_CXPB, mutpb=SIMPLE_MUTPB, ngen=NGEN, stats=stats, verbose=False)

def muPlusLambda(pop):
    MU_PLUS_LAMBDA_MUTPB = 0.2
    MU_PLUS_LAMBDA_CXPB = 0.7

    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)
    toolbox.register("mate", tools.cxTwoPoint)
    return algorithms.eaMuPlusLambda(population=pop, toolbox=toolbox, mu=12, lambda_=24, cxpb=MU_PLUS_LAMBDA_CXPB, mutpb=MU_PLUS_LAMBDA_MUTPB, ngen=NGEN, stats=stats, halloffame=hof2, verbose=False)

def muCommaLambda(pop):
    MU_COMMA_LAMBDA_MUTPB = 0.4
    MU_COMMA_LAMBDA_CXPB = 0.5

    toolbox.register("select", tools.selTournament, tournsize=10)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("mate", tools.cxOnePoint)
    return algorithms.eaMuCommaLambda(population=pop, toolbox=toolbox, mu=5, lambda_=10, cxpb=MU_COMMA_LAMBDA_CXPB, mutpb=MU_COMMA_LAMBDA_MUTPB, ngen=NGEN, stats=stats, halloffame=hof3, verbose=False)

for i in range(0, EXPERIMENTS):
    results1, log1 = simple(toolbox.population(n=10))
    toolbox.unregister('select')
    toolbox.unregister('mutate')
    toolbox.unregister('mate')

    results2, log2 = muPlusLambda(toolbox.population(n=10))
    toolbox.unregister('select')
    toolbox.unregister('mutate')
    toolbox.unregister('mate')

    results3, log3 = muCommaLambda(toolbox.population(n=10))
    toolbox.unregister('select')
    toolbox.unregister('mutate')
    toolbox.unregister('mate')

    df1 = pd.DataFrame({
        'experiment': [i] * (NGEN + 1),
        'iteration':list(range(0, NGEN + 1)),
        'algorithm': ['eaSimple'] * (NGEN + 1),
        'max': list(log1.select('max')),
    })
    df2 = pd.DataFrame({
        'experiment': [i] * (NGEN + 1),
        'iteration':list(range(0, NGEN + 1)),
        'algorithm': ['eaMuPlusLambda'] * (NGEN + 1),
        'max': list(log2.select('max')),
    })
    df3 = pd.DataFrame({
        'experiment': [i] * (NGEN + 1),
        'iteration':list(range(0, NGEN + 1)),
        'algorithm': ['eaMuCommaLambda'] * (NGEN + 1),
        'max': list(log3.select('max')),
    })

    for df in [df1, df2, df3]:
        df.at[0, 'greater'] = df.loc[0]['max']
        df.loc[1:].apply(lambda row: greater(row), axis=1)

    eaSimple = eaSimple.append(df1)
    eaMuPlusLambda = eaMuPlusLambda.append(df2)
    eaMuCommaLambda = eaMuCommaLambda.append(df3)

for df in [eaSimple, eaMuPlusLambda, eaMuCommaLambda]:
    df.reset_index(drop=True, inplace=True)

pd.set_option("display.max_rows", None, "display.max_columns", None)
print(eaSimple)
print(eaMuPlusLambda)
print(eaMuCommaLambda)

figure, plts = plt.subplots(1, 3)
graph = 0
for df in [eaSimple, eaMuPlusLambda, eaMuCommaLambda]:
    results = df.groupby('iteration').agg({'max': ['mean', 'std']})
    print(results)
    maximums = results['max']['mean'].values
    std = results['max']['std'].values
    plts[graph].plot(range(0,NGEN + 1), maximums, color='red', marker='*')
    plts[graph].plot(range(0,NGEN + 1), maximums+std, color='b', linestyle='-.')
    plts[graph].plot(range(0,NGEN + 1), maximums-std, color='b', marker='o')
    plts[graph].set(xlabel='gen', ylabel='profit')
    plts[graph].legend(['promedio', 'promedio + std','promedio - std'])
    plts[graph].set_title(df.at[1, 'algorithm'])    
    graph += 1

[print('eaSimple', func_eval(ind)[0], np.sum(np.asarray(ind) * np.asarray(w['weights'].values.tolist()))) for ind in hof1]
[print('eaMuPlusLambda', func_eval(ind)[0], np.sum(np.asarray(ind) * np.asarray(w['weights'].values.tolist()))) for ind in hof2]
[print('eaMuCommaLambda', func_eval(ind)[0], np.sum(np.asarray(ind) * np.asarray(w['weights'].values.tolist()))) for ind in hof3]

#plt.tight_layout()
plt.show()