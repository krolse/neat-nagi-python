from nagi.constants import FLIP_POINT, NUM_TIME_STEPS
from nagi.simulation import Environment, Agent
from nagi.visualization import visualize_genome
import matplotlib.pyplot as plt

import pickle

with open('../data/test_genome.pkl', 'rb') as file:
    test_genome = pickle.load(file)

for i in [4, 6, 7]:
    test_genome.nodes[i].stdp_parameters['std_minus'] = 20
    test_genome.nodes[i].stdp_parameters['std_plus'] = 10
    test_genome.nodes[i].stdp_parameters['a_plus'] = 5
    test_genome.nodes[i].stdp_parameters['a_minus'] = 20

    print(test_genome.nodes[i].stdp_parameters)

agent = Agent.create_agent(test_genome)
agent.spiking_neural_network.neurons[7].inputs[6] = 0.2
for neuron in agent.spiking_neural_network.neurons.values():
    neuron.bias = 0
visualize_genome(test_genome, True)
environment = Environment(50, 5)
_, fitness, weights, membrane_potentials, time_step = environment.simulate_with_visualization(agent)

t_values = range(time_step + 1)
number_of_neurons = len(membrane_potentials.keys())
number_of_weights = len(weights.keys())

# Membrane potential
fig = plt.figure()
plt.title("Neuron membrane potentials")
for i, key in enumerate(membrane_potentials.keys()):
    plt.subplot(number_of_neurons, 1, i + 1)
    plt.ylabel(f"{key}")
    plt.xlabel("Time (in ms)")
    plt.plot(t_values, [membrane_potential[0] for membrane_potential in membrane_potentials[key]], 'g-')
    plt.plot(t_values, [membrane_potential[1] for membrane_potential in membrane_potentials[key]], 'r-')
    flip_points = [i for i in range(len(membrane_potentials[key])) if
                   i >= FLIP_POINT * NUM_TIME_STEPS and i % (FLIP_POINT * NUM_TIME_STEPS) == 0]
    for flip_point in flip_points:
        plt.axvline(x=flip_point)

# Weights
fig = plt.figure()
plt.title("Weights")
for i, key in enumerate(weights.keys()):
    plt.subplot(number_of_weights, 1, i + 1)
    plt.ylabel(f"{key}")
    plt.xlabel("Time (in ms)")
    plt.plot(t_values, weights[key], 'b-')
    flip_points = [i for i in range(len(weights[key])) if
                   i >= FLIP_POINT * NUM_TIME_STEPS and i % (FLIP_POINT * NUM_TIME_STEPS) == 0]
    for flip_point in flip_points:
        plt.axvline(x=flip_point)

plt.show()
print(f'Fitness: {fitness}')
