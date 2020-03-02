from nagi.simulation import Environment, Agent
import pickle

with open('../data/test_genome.pkl', 'rb') as file:
    test_genome = pickle.load(file)
agent = Agent.create_agent(test_genome)
for neuron in agent.spiking_neural_network.neurons.values():
    neuron.bias = 0
    for key, weight in neuron.inputs.items():
        print(f'{weight}')
environment = Environment(100, 5)
fitness = environment.simulate(agent)
print(f'Fitness: {fitness[1]}')
