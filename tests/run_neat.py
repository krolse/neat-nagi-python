import pickle

from nagi.neat import Population
from nagi.simulation import Environment, Agent
import multiprocessing as mp

if __name__ == '__main__':
    pop = Population(100, 4, 2)
    max_fitnesses = []
    average_fitnesses = []
    pool = mp.Pool(mp.cpu_count())

    for i in range(10):
        print(f'Generation {i}...')
        env = Environment(300, 5)
        results = pool.map(env.simulate, [Agent.create_agent(genome) for genome in pop.genomes.values()])
        fitnesses = {result[0]: result[1] for result in results}

        max_fitnesses.append(max(fitnesses.values()))
        average_fitnesses.append(sum(fitnesses.values()) / len(fitnesses))

        pop.next_generation(fitnesses)
        with open('../data/test_run.pkl', 'wb') as file:
            pickle.dump((average_fitnesses, max_fitnesses), file)
