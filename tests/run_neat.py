import multiprocessing as mp
import os
import pickle
from copy import deepcopy

import tqdm

from nagi.neat import Population
from nagi.simulation_1d import OneDimensionalEnvironment, OneDimensionalAgent


def get_file_path():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_number = 1
    while os.path.exists(root_path + f'/data/test_run_{run_number}.pkl'):
        run_number += 1
    return root_path + f'/data/test_run_{run_number}.pkl'


if __name__ == '__main__':
    path = get_file_path()
    pool = mp.Pool(mp.cpu_count())

    population = Population(50, 4, 2)
    generations = {}
    for i in range(0, 100):
        print(f'\nGeneration {i}...')
        env = OneDimensionalEnvironment(50, 5)
        agents = list([OneDimensionalAgent.create_agent(genome) for genome in population.genomes.values()])
        results = tqdm.tqdm(pool.imap_unordered(env.simulate, agents), total=(len(agents)))
        fitnesses = {result[0]: result[1] for result in results}
        generations[i] = {'population': deepcopy(population), 'fitnesses': fitnesses}

        with open(path, 'wb') as file:
            pickle.dump(generations, file)

        population.next_generation(fitnesses)
