import math
import pickle
import sys
from typing import Dict
from pathlib import Path

from easygui import fileopenbox

from definitions import ROOT_PATH


def get_nth_top_genome(n: int, measure: str, results: Dict[int, Dict]):
    individuals = [(generation_number, genome_key) for (generation_number, value) in results.items() for genome_key in
                   value['population'].genomes.keys()]

    if measure == 'fitness':
        individuals.sort(key=lambda x: results[x[0]]['fitnesses'][x[1]], reverse=True)
    elif measure == 'accuracy':
        individuals.sort(key=lambda x: results[x[0]]['accuracies'][x[1]], reverse=True)
    elif measure == 'end_of_sample_accuracy':
        individuals.sort(key=lambda x: (results[x[0]]['end_of_sample_accuracies'][x[1]],
                                        results[x[0]]['accuracies'][x[1]]), reverse=True)
    else:
        raise Exception("Error: Faulty measure.")
    generation, genome_id = individuals[n]
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
    i = int(input("Select measure (Fitness = 1, Accuracy = 2, End of sample accuracy = 3): "))
    try:
        measure = {1: 'fitness', 2: 'accuracy', 3: 'end_of_sample_accuracy'}[i]
    except KeyError("Error: Faulty input (must be 1, 2 or 3)."):
        sys.exit(-1)
    n = int(input(f"Select which nth best individual to extract: "))
    if n > len(data):
        raise IndexError("Error: n is bigger than population size")
    genome = get_nth_top_genome(n, measure, data)
    genome_path = f'{ROOT_PATH}/data/{Path(path).stem}_{ordinal(n)}_most_{shorthand_measure(measure)}_genome.pkl'
    with open(genome_path, 'wb') as file:
        pickle.dump(genome, file)
