import pickle
from functools import reduce
from easygui import fileopenbox

import numpy as np
import matplotlib.pyplot as plt
from definitions import ROOT_PATH

path = fileopenbox(default=f"{ROOT_PATH}/data/*test_run*.pkl")
with open(path, 'rb') as file:
    data = pickle.load(file)

num_species = max(reduce(lambda a, b: a + b,
                         [[key for key in generation['population'].species.keys()]
                          for generation in data.values()])) + 1

x = range(len(data))
m = [[0 if i not in generation['population'].species else len(generation['population'].species[i]) for generation in data.values()] for i in range(num_species)]
y = np.vstack(m)
labels = [f's{i}' for i in range(num_species)]
fig = plt.figure()
plt.title("Species distribution")
plt.stackplot(x, y)
plt.legend(labels=labels)
plt.margins(0)
plt.show()
