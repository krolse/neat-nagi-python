import multiprocessing as mp
import pickle
from copy import deepcopy

import tqdm

from nagi.neat import Population
from nagi.simulation_1d import OneDimensionalEnvironment, OneDimensionalAgent

if __name__ == '__main__':
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

        with open('../data/test_run_6.pkl', 'wb') as file:
            pickle.dump(generations, file)

        population.next_generation(fitnesses)
