import pickle
import sys
from typing import List

from nagi import constants
from nagi.constants import TIME_STEP_IN_MSEC
from nagi.snn import SpikingNeuralNetwork


def generate_spike_train(frequency: int, number_of_seconds: int, time_step_in_msec: float) -> List[int]:
    number_of_steps = int(number_of_seconds / (time_step_in_msec / 1000))
    nth = int(number_of_steps / frequency)
    return [40 if i % nth == 0 else 0 for i in range(number_of_steps)]


def generate_spike_frequency(frequency: int, time_step_in_msec: float) -> int:
    return int(1 / (time_step_in_msec / 1000) / frequency)


def get_reward_frequency(sample: int, eat: int, avoid: int, current_frequency: int):
    if sample:
        if eat > avoid:
            return high_frequency
        elif eat < avoid:
            return low_frequency
        else:
            return current_frequency
    else:
        if eat > avoid:
            return low_frequency
        elif eat < avoid:
            return high_frequency
        else:
            return current_frequency


def swap_frequency(frequency_1):
    return high_frequency if frequency_1 == low_frequency else low_frequency


def sum_of_spikes(counter):
    return sum([value for key, value in counter.items() if time_step - key < max_counter_length])


high_frequency = generate_spike_frequency(500, TIME_STEP_IN_MSEC)
low_frequency = generate_spike_frequency(5, TIME_STEP_IN_MSEC)
spike_actuator_window = 0.5
max_counter_length = int(spike_actuator_window / (TIME_STEP_IN_MSEC / 1000))
food = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0]

with open('../data/test_genome.pkl', 'rb') as file:
    test_genome = pickle.load(file)
# visualize_genome(test_genome)
snn = SpikingNeuralNetwork.create(test_genome, 5.0, **constants.REGULAR_SPIKING_PARAMS)
for sample_food in food:
    eat_counter = {}
    avoid_counter = {}
    input_frequency_1 = high_frequency if sample_food else low_frequency
    input_frequency_2 = low_frequency if sample_food else high_frequency
    reward_frequency_1 = sys.maxsize
    reward_frequency_2 = sys.maxsize
    for time_step in range(25000):
        reward_frequency_1 = get_reward_frequency(sample_food, sum_of_spikes(eat_counter), sum_of_spikes(avoid_counter),
                                                  reward_frequency_1)
        reward_frequency_2 = low_frequency if reward_frequency_1 == high_frequency else high_frequency
        inputs = [40 if time_step % input_frequency_1 == 0 else 0,
                  40 if time_step % input_frequency_2 == 0 else 0,
                  40 if time_step % reward_frequency_1 == 0 else 0,
                  40 if time_step % reward_frequency_2 == 0 else 0]
        snn.set_inputs(inputs)
        eat, avoid = snn.advance(TIME_STEP_IN_MSEC)
        if eat:
            eat_counter[time_step] = 1
        if avoid:
            avoid_counter[time_step] = 1
    print(f'Food: {sample_food}, Eat: {sum_of_spikes(eat_counter)}, Avoid: {sum_of_spikes(avoid_counter)}')
