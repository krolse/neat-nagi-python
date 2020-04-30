import pickle

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from easygui import fileopenbox

from definitions import ROOT_PATH
from nagi.constants import FLIP_POINT_1D, NUM_TIME_STEPS, RED, BLUE, GREEN
from nagi.simulation_2d import TwoDimensionalEnvironment, TwoDimensionalAgent
from nagi.visualization import visualize_genome

with open(f'{fileopenbox(default=f"{ROOT_PATH}/data/*genome*.pkl")}', 'rb') as file:
    test_genome = pickle.load(file)

agent = TwoDimensionalAgent.create_agent(test_genome)
visualize_genome(test_genome, True)
environment = TwoDimensionalEnvironment(50, 5, testing=True)
(_,
 fitness,
 weights,
 membrane_potentials,
 time_step,
 intervals,
 actuators,
 accuracy,
 end_of_sample_accuracy,
 samples,
 logic_gates) = environment.simulate_with_visualization(agent)

number_of_neurons = len(membrane_potentials.keys())
number_of_weights = len(weights.keys())
t_values = range(time_step + 1)
alpha = 0.85


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


def add_fig_legend(*args):
    colors = [arg[0] for arg in args]
    labels = [arg[1] for arg in args]
    plt.figlegend((*(Line2D([0], [0], color=color) for color in colors),
                   Line2D([0], [0], color='gray', linestyle='--'),
                   Line2D([0], [0], color='k'),
                   Patch(color=RED)),
                  (*(label for label in labels), 'sample change', 'flip point', 'wrong region'),
                  loc='upper left')


def add_table():
    the_table = plt.table(cellText=[[sample for sample in samples],
                                    [food.name for food in logic_gates]],
                          rowLabels=['Food sample', 'Beneficial food'],
                          cellLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)


# Membrane potential
fig = plt.figure()
fig.suptitle("Neuron membrane potentials")
for i, key in enumerate(sorted(membrane_potentials.keys())):
    plt.subplot(number_of_neurons, 1, i + 1)
    plt.ylabel(f"{key}")
    if i == 0:
        plt.xlabel("Time (in ms)")
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
    else:
        plt.gca().axes.xaxis.set_visible(False)
    plt.plot(t_values, [membrane_potential[0] for membrane_potential in membrane_potentials[key]],
             color=GREEN, linestyle='-', alpha=alpha)
    plt.plot(t_values, [membrane_potential[1] for membrane_potential in membrane_potentials[key]],
             color=BLUE, linestyle='-', alpha=alpha)
    plt.xlim(0, len(t_values) + NUM_TIME_STEPS - len(t_values) % NUM_TIME_STEPS)
    add_fig_legend((GREEN, 'membrane potential'), (BLUE, 'membrane threshold'))
    add_vertical_lines_and_background(4)
add_table()

# Weights
fig = plt.figure()
plt.suptitle("Weights")
for i, key in enumerate(sorted(weights.keys(), key=lambda x: x[1])):
    plt.subplot(number_of_weights, 1, i + 1)
    plt.ylabel(f"{key}")
    if i == 0:
        plt.xlabel("Time (in ms)")
        plt.gca().xaxis.set_ticks_position('top')
        plt.gca().xaxis.set_label_position('top')
    else:
        plt.gca().axes.xaxis.set_visible(False)
    plt.plot(t_values, weights[key], color=BLUE, linestyle='-')
    add_fig_legend((BLUE, 'weight'))
    add_vertical_lines_and_background(2)
    plt.xlim(0, len(t_values) + NUM_TIME_STEPS - len(t_values) % NUM_TIME_STEPS)
add_table()

# Actuator history
fig = plt.figure()
fig.suptitle("Actuator history")
eat_actuators = [actuator[0] for actuator in actuators]
avoid_actuators = [actuator[1] for actuator in actuators]
plt.plot(t_values, eat_actuators, color=GREEN, alpha=alpha)
plt.plot(t_values, avoid_actuators, color=BLUE, alpha=alpha)
plt.xlabel("Time (in ms)")
plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
add_fig_legend((GREEN, 'eat actuator'), (BLUE, 'avoid actuator'))
add_vertical_lines_and_background(max(max(eat_actuators), max(avoid_actuators)) + 2)
add_table()
plt.xlim(0, len(t_values) + NUM_TIME_STEPS - len(t_values) % NUM_TIME_STEPS)

print(f'\n **** Results ****')
print(f'Fitness: {fitness:.3f}')
print(f'Accuracy: {accuracy * 100:.1f}%')
print(f'End of sample accuracy: {end_of_sample_accuracy * 100:.1f}%')
plt.show()
