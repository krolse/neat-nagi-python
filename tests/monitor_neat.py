import pickle

from easygui import fileopenbox
import matplotlib.pyplot as plt
from definitions import ROOT_PATH

if __name__ == '__main__':
    path = fileopenbox(default=f"{ROOT_PATH}/data/*test_run*.pkl")
    with open(path, 'rb') as file:
        data = pickle.load(file)

    fitnesses = [len(generation['fitnesses']) for generation in data.values()]
    species = [len(generation['population'].species) for generation in data.values()]
    genomes = [len(generation['population'].genomes) for generation in data.values()]
    fig = plt.figure()
    x = range(len(fitnesses))
    plt.plot(x, fitnesses, label='fitnesses')
    plt.plot(x, species, label='species')
    plt.plot(x, genomes, label='population size')
    plt.figlegend()
    plt.show()
