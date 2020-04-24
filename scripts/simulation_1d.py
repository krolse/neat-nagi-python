import pickle

import matplotlib.pyplot as plt
from easygui import fileopenbox

from definitions import ROOT_PATH
from nagi.constants import FLIP_POINT_1D, NUM_TIME_STEPS, RED, BLUE, GREEN
from nagi.simulation_1d import OneDimensionalEnvironment, OneDimensionalAgent
from nagi.visualization import visualize_genome

with open(f'{fileopenbox(default=f"{ROOT_PATH}/data/*genome*.pkl")}', 'rb') as file:
    test_genome = pickle.load(file)

agent = OneDimensionalAgent.create_agent(test_genome)
visualize_genome(test_genome, True)
environment = OneDimensionalEnvironment(50, 5)
(_,
 fitness,
 weights,
 membrane_potentials,
 time_step,
 intervals,
 actuators,
 accuracy,
 end_of_sample_accuracy) = environment.simulate_with_visualization(agent)

number_of_neurons = len(membrane_potentials.keys())
number_of_weights = len(weights.keys())
t_values = range(time_step + 1)


def add_vertical_lines_and_background(height: int):
    flip_points = [flip_point for flip_point in range(len(t_values))
                   if flip_point >= FLIP_POINT_1D * NUM_TIME_STEPS
                   and flip_point % (FLIP_POINT_1D * NUM_TIME_STEPS) == 0]
    sample_points = [sample_point for sample_point in range(len(t_values)) if
                     sample_point >= NUM_TIME_STEPS and
                     sample_point % NUM_TIME_STEPS == 0 and
                     sample_point not in flip_points]

    for flip_point in flip_points:
        plt.axvline(x=flip_point, color='k')
    for sample_point in sample_points:
        plt.axvline(x=sample_point, color='gray', linestyle='--')
    for start, end in intervals:
        rect = plt.Rectangle((start, 0), end - start, height=height, facecolor=RED)
        plt.gca().add_patch(rect)


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
    add_vertical_lines_and_background(4)

# Weights
fig = plt.figure()
plt.title("Weights")
for i, key in enumerate(sorted(weights.keys(), key=lambda x: x[1])):
    plt.subplot(number_of_weights, 1, i + 1)
    plt.ylabel(f"{key}")
    plt.xlabel("Time (in ms)")
    plt.plot(t_values, weights[key], color=BLUE, linestyle='-')
    add_vertical_lines_and_background(2)

# Actuator history
fig = plt.figure()
plt.title("Actuator history")
eat_actuators = [actuator[0] for actuator in actuators]
avoid_actuators = [actuator[1] for actuator in actuators]
plt.plot(t_values, eat_actuators, color=GREEN)
plt.plot(t_values, avoid_actuators, color=BLUE)
add_vertical_lines_and_background(max(max(eat_actuators), max(avoid_actuators)) + 2)
print(f'\n **** Results ****')
print(f'Fitness: {fitness:.3f}')
print(f'Accuracy: {accuracy * 100:.1f}%')
print(f'End of sample accuracy: {end_of_sample_accuracy*100:.1f}%')
plt.show()
