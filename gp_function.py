from deap import base, creator, algorithms, tools, gp
import numpy as np

x = np.array(
     [(0, 10), (1, 9), (2, 8), (3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (8, 2), (9, 1)]
)

y = np.array(
    [90, 82, 74, 66, 58, 50, 42, 34, 26, 18] 
)

