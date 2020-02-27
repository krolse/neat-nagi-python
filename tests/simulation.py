from nagi.simulation import Environment, Agent
import pickle

with open('../data/test_genome.pkl', 'rb') as file:
    test_genome = pickle.load(file)
environment = Environment(200, 5)
fitness = environment.simulate(Agent.create_agent(test_genome))
print(f'Fitness: {fitness[1]}')
