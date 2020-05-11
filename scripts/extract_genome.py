import math
import pickle
import re
import sys
from typing import Dict

from easygui import fileopenbox

from definitions import ROOT_PATH


def get_nth_top_genome(n: int, measure: str, results: Dict[int, Dict]):
    if measure == 'fitness':
        top_individuals = {key: max(value['fitnesses'].items(),
                                    key=lambda i: i[1]) for key, value in results.items()}
    elif measure == 'accuracy':
        top_individuals = {key: max(value['accuracies'].items(),
                                    key=lambda i: i[1]) for key, value in results.items()}
    elif measure == 'end_of_sample_accuracies':
        top_individuals = {key: max(value['end_of_sample_accuracies'].items(),
                                    key=lambda i: i[1]) for key, value in results.items()}
    else:
        raise Exception("Error: Faulty measure.")

    individual = sorted([(key, value) for key, value in top_individuals.items()],
                        key=lambda i: i[1][1], reverse=True)[n - 1]
    generation = individual[0]
    genome_id = individual[1][0]

    return results[generation]['population'].genomes[genome_id]


def ordinal(n):
    return f'{n}{"tsnrhtdd"[(math.floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10::4]}'


def shorthand_measure(measure: str):
    if measure == 'fitness':
        return 'fit'
    if measure == 'accuracy':
        return 'acc'
    if measure == 'end_of_sample_accuracy':
        return 'eos_acc'


if __name__ == '__main__':
    path = fileopenbox(default=f"{ROOT_PATH}/data/*test_run*.pkl")
    with open(path, 'rb') as file:
        data = pickle.load(file)
    run_datetime = re.search(r'[0-9]{8}-[0-9]{6}', path).group()
    i = int(input("Select measure (Fitness = 1, Accuracy = 2, End of sample accuracy = 3): "))
    try:
        measure = {1: 'fitness', 2: 'accuracy', 3: 'end_of_sample_accuracy'}[i]
    except KeyError("Error: Faulty input (must be 1, 2 or 3)."):
        sys.exit(-1)
    n = int(input(f"Select which nth best individual to extract (a number between 1 and {len(data)}): "))
    if n > len(data):
        raise IndexError("Error: n is bigger than population size")
    genome = get_nth_top_genome(n, measure, data)
    genome_path = f'{ROOT_PATH}/data/test_run_{run_datetime}_{ordinal(n)}_most_{shorthand_measure(measure)}_genome.pkl'
    with open(genome_path, 'wb') as file:
        pickle.dump(genome, file)
