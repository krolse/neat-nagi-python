import pickle
from typing import Dict

import matplotlib.pyplot as plt
from easygui import fileopenbox

from definitions import ROOT_PATH


def get_most_fit_genome(results: Dict[int, Dict]):
    most_fit_individuals = {key: max(value['fitnesses'].items(), key=lambda i: i[1]) for key, value in results.items()}
    most_fit_individual = max([(key, value) for key, value in most_fit_individuals.items()], key=lambda i: i[1][1])
    generation = most_fit_individual[0]
    genome_id = most_fit_individual[1][0]

    return results[generation]['population'].genomes[genome_id]


if __name__ == '__main__':
    with open(f'{fileopenbox(default=f"{ROOT_PATH}/data/test_run*.pkl")}', 'rb') as file:
        data = pickle.load(file)

    # genome = get_most_fit_genome(data)
    # with open('../data/most_fit_genome_big.pkl.pkl', 'wb') as file:
    #     pickle.dump(genome, file)
    # visualize_genome(genome)

    fitnesses = [generation['fitnesses'] for generation in data.values()]
    average_fitnesses = [sum(generation.values()) / len(generation) for generation in fitnesses]
    # max_fitnesses = [max(generation.values()) for generation in fitnesses]

    x = range(len(average_fitnesses))
    fig = plt.figure()
    plt.ylim(0, 1)
    plt.ylabel('fitness')
    plt.xlabel('generation')
    for generation, fitness in enumerate([fitness.values() for fitness in fitnesses]):
        plt.scatter([generation for _ in range(len(fitness))], fitness, color='b', s=0.1)
    plt.plot(x, average_fitnesses, 'r')
    plt.show()
