from nagi.constants import FLIP_POINT_1D, NUM_TIME_STEPS
from nagi.simulation_2d import TwoDimensionalEnvironment, TwoDimensionalAgent
from nagi.visualization import visualize_genome
import matplotlib.pyplot as plt

import pickle

with open('../data/test_genome_2d.pkl', 'rb') as file:
    test_genome = pickle.load(file)

agent = TwoDimensionalAgent.create_agent(test_genome)
for neuron in agent.spiking_neural_network.neurons.values():
    neuron.bias = 0
visualize_genome(test_genome, True)
environment = TwoDimensionalEnvironment(50, 5)
_, fitness, weights, membrane_potentials, time_step = environment.simulate_with_visualization(agent)

t_values = range(time_step + 1)
number_of_neurons = len(membrane_potentials.keys())
number_of_weights = len(weights.keys())

# Membrane potential
fig = plt.figure()
plt.title("Neuron membrane potentials")
for i, key in enumerate(sorted(membrane_potentials.keys())):
    plt.subplot(number_of_neurons, 1, i + 1)
    plt.ylabel(f"{key}")
    plt.xlabel("Time (in ms)")
    plt.plot(t_values, [membrane_potential[0] for membrane_potential in membrane_potentials[key]], 'g-')
    plt.plot(t_values, [membrane_potential[1] for membrane_potential in membrane_potentials[key]], 'r-')
    flip_points = [i for i in range(len(membrane_potentials[key])) if
                   i >= FLIP_POINT_1D * NUM_TIME_STEPS and i % (FLIP_POINT_1D * NUM_TIME_STEPS) == 0]
    for flip_point in flip_points:
        plt.axvline(x=flip_point)

# Weights
fig = plt.figure()
plt.title("Weights")
for i, key in enumerate(sorted(weights.keys(), key=lambda x: x[1])):
    plt.subplot(number_of_weights, 1, i + 1)
    plt.ylabel(f"{key}")
    plt.xlabel("Time (in ms)")
    plt.plot(t_values, weights[key], 'b-')
    flip_points = [i for i in range(len(weights[key])) if
                   i >= FLIP_POINT_1D * NUM_TIME_STEPS and i % (FLIP_POINT_1D * NUM_TIME_STEPS) == 0]
    for flip_point in flip_points:
        plt.axvline(x=flip_point)

print(f'Fitness: {fitness}')
plt.show()
