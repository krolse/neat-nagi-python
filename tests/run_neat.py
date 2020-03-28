import pickle
import tqdm

from nagi.neat import Population
from nagi.simulation_1d import OneDimensionalEnvironment, OneDimensionalAgent
import multiprocessing as mp

if __name__ == '__main__':
    pop = Population(50, 4, 2)
    max_fitnesses = []
    average_fitnesses = []
    pool = mp.Pool(mp.cpu_count())

    for i in range(20):
        print(f'\nGeneration {i}...')
        env = OneDimensionalEnvironment(50, 5)
        agents = list([OneDimensionalAgent.create_agent(genome) for genome in pop.genomes.values()])
        results = tqdm.tqdm(pool.imap_unordered(env.simulate, agents), total=(len(agents)))
        fitnesses = {result[0]: result[1] for result in results}

        max_fitnesses.append(max(fitnesses.values()))
        average_fitnesses.append(sum(fitnesses.values()) / len(fitnesses))

        with open('../data/test_run.pkl', 'wb') as file:
            pickle.dump((average_fitnesses, max_fitnesses), file)

        pop.next_generation(fitnesses)

