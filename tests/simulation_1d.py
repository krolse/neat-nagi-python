from nagi.constants import FLIP_POINT_1D, NUM_TIME_STEPS, RED, BLUE, GREEN
from nagi.simulation_1d import OneDimensionalEnvironment, OneDimensionalAgent
from nagi.visualization import visualize_genome
import matplotlib.pyplot as plt

import pickle

with open('../data/most_fit_genome_test_run_5.pkl', 'rb') as file:
    test_genome = pickle.load(file)

agent = OneDimensionalAgent.create_agent(test_genome)
visualize_genome(test_genome, True)
environment = OneDimensionalEnvironment(50, 5)
_, fitness, weights, membrane_potentials, time_step, intervals = environment.simulate_with_visualization(agent)

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
    plt.plot(t_values, [membrane_potential[0] for membrane_potential in membrane_potentials[key]],
             color=GREEN, linestyle='-')
    plt.plot(t_values, [membrane_potential[1] for membrane_potential in membrane_potentials[key]],
             color=BLUE, linestyle='-')
    flip_points = [i for i in range(len(membrane_potentials[key])) if
                   i >= FLIP_POINT_1D * NUM_TIME_STEPS and i % (FLIP_POINT_1D * NUM_TIME_STEPS) == 0]
    sample_points = [i for i in range(len(membrane_potentials[key])) if
                     i >= NUM_TIME_STEPS and
                     i % NUM_TIME_STEPS == 0 and
                     i not in flip_points]

    for flip_point in flip_points:
        plt.axvline(x=flip_point, color='k')
    for sample_point in sample_points:
        plt.axvline(x=sample_point, color='gray', linestyle='--')
    for start, end in intervals:
        height = 4
        rect = plt.Rectangle((start, 0), end - start, height=height, facecolor=RED)
        plt.gca().add_patch(rect)


# Weights
fig = plt.figure()
plt.title("Weights")
for i, key in enumerate(sorted(weights.keys(), key=lambda x: x[1])):
    plt.subplot(number_of_weights, 1, i + 1)
    plt.ylabel(f"{key}")
    plt.xlabel("Time (in ms)")
    plt.plot(t_values, weights[key], color=BLUE, linestyle='-')
    flip_points = [i for i in range(len(weights[key])) if
                   i >= FLIP_POINT_1D * NUM_TIME_STEPS and i % (FLIP_POINT_1D * NUM_TIME_STEPS) == 0]
    sample_points = [i for i in range(len(weights[key])) if
                     i >= NUM_TIME_STEPS and
                     i % NUM_TIME_STEPS == 0 and
                     i not in flip_points]

    for flip_point in flip_points:
        plt.axvline(x=flip_point, color='k')
    for sample_point in sample_points:
        plt.axvline(x=sample_point, color='gray', linestyle='--')
    for start, end in intervals:
        height = 2
        rect = plt.Rectangle((start, 0), end - start, height=height, facecolor="red", alpha=0.5)
        plt.gca().add_patch(rect)

print(f'Fitness: {fitness}')
plt.show()
