import multiprocessing as mp
import os
import pickle
from copy import deepcopy

import tqdm

from definitions import ROOT_PATH
from nagi.neat import Population
from nagi.simulation_1d import OneDimensionalEnvironment, OneDimensionalAgent


def get_file_paths():
    run_number = 1
    while os.path.exists(f'{ROOT_PATH}/data/test_run_{run_number}.pkl'):
        run_number += 1
    return f'{ROOT_PATH}/data/test_run_{run_number}.pkl', f'{ROOT_PATH}/data/test_run_{run_number}_config.txt'


def generate_config_string():
    with open(f'{ROOT_PATH}/nagi/constants.py', 'r') as f:
        constants = f.read()

    return f"""Environment type: {environment_type}
Input size: {input_size}
Output size: {output_size}
Population size: {population_size}
Number of generations: {number_of_generations}
High frequency: {high_frequency}
Low frequency: {low_frequency}

{constants}"""


environment_type = "1D"
input_size, output_size = 4, 2
high_frequency = 50
low_frequency = 5

population_size = 200
number_of_generations = 100


if __name__ == '__main__':
    pickle_path, txt_path = get_file_paths()

    with open(txt_path, 'w') as file:
        file.write(generate_config_string())

    pool = mp.Pool(mp.cpu_count())

    population = Population(population_size, input_size, output_size)
    generations = {}
    for i in range(0, number_of_generations):
        print(f'\nGeneration {i}...')
        env = OneDimensionalEnvironment(high_frequency, low_frequency)
        agents = list([OneDimensionalAgent.create_agent(genome) for genome in population.genomes.values()])
        results = tqdm.tqdm(pool.imap_unordered(env.simulate, agents), total=(len(agents)))
        fitnesses = {result[0]: result[1] for result in results}
        accuracies = {result[0]: result[2] for result in results}
        end_of_sample_accuracies = {result[0]: result[3] for result in results}
        highest_fitness = max(fitnesses.values())
        print(f'Highest fitness: {highest_fitness:.3f}')
        generations[i] = {'population': deepcopy(population),
                          'fitnesses': fitnesses,
                          'accuracies': accuracies,
                          'end_of_sample_accuracies': end_of_sample_accuracies}

        with open(pickle_path, 'wb') as file:
            pickle.dump(generations, file)

        population.next_generation(fitnesses)
