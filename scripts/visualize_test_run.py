import pickle
from typing import Dict

import matplotlib.pyplot as plt
from easygui import fileopenbox
from matplotlib.lines import Line2D

from definitions import ROOT_PATH
from nagi.constants import RED, BLUE, GREEN, GOLD, ORANGE


def get_most_fit_genome(results: Dict[int, Dict]):
    most_fit_individuals = {key: max(value['fitnesses'].items(), key=lambda i: i[1]) for key, value in results.items()}
    most_fit_individual = max([(key, value) for key, value in most_fit_individuals.items()], key=lambda i: i[1][1])
    generation = most_fit_individual[0]
    genome_id = most_fit_individual[1][0]

    return results[generation]['population'].genomes[genome_id]


if __name__ == '__main__':
    path = fileopenbox(default=f"{ROOT_PATH}/data/test_run*.pkl")
    with open(path, 'rb') as file:
        data = pickle.load(file)

    genome = get_most_fit_genome(data)
    run_number = path[path.find('.pkl') - 1]
    with open(f'{ROOT_PATH}/data/most_fit_genome_test_run_{run_number}.pkl', 'wb') as file:
        pickle.dump(genome, file)

    fitnesses = [generation['fitnesses'] for generation in data.values()]
    average_fitnesses = [sum(generation.values()) / len(generation) for generation in fitnesses]

    x = range(len(average_fitnesses))
    fig = plt.figure()
    plt.ylim(0, 1)
    plt.ylabel('fitness')
    plt.xlabel('generation')
    for generation, fitness in enumerate([fitness.values() for fitness in fitnesses]):
        plt.scatter([generation for _ in range(len(fitness))], fitness, color='b', s=0.1)
    plt.plot(x, average_fitnesses, RED)

    try:
        test_fitnesses = [generation['test_fitness'] for generation in data.values()]
        plt.plot(x, test_fitnesses, GREEN)
    except KeyError:  # 1D environments don't have test fitnesses
        pass

    try:
        accuracies = [max(generation['accuracies'].values()) for generation in data.values()]
        end_of_sample_accuracies = [max(generation['end_of_sample_accuracies'].values()) for generation in data.values()]
        plt.plot(x, accuracies, GOLD)
        plt.plot(x, end_of_sample_accuracies, ORANGE)
    except KeyError:  # Older test run data don't contain accuracies
        pass

    plt.figlegend((Line2D([0], [0], marker='o', color='b', linewidth=0),
                   Line2D([0], [0], color=RED),
                   Line2D([0], [0], color=GREEN),
                   Line2D([0], [0], color=GOLD),
                   Line2D([0], [0], color=ORANGE)),
                  ('fitness',
                   'average fitness',
                   'max test fitness',
                   'max accuracy',
                   'max eos accuracy'),
                  loc='upper left')
    plt.show()
