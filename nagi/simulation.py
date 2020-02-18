import pickle
import random
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


high_frequency = generate_spike_frequency(300, TIME_STEP_IN_MSEC)
low_frequency = generate_spike_frequency(5, TIME_STEP_IN_MSEC)
food = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0]

with open('../data/test_genome.pkl', 'rb') as file:
    test_genome = pickle.load(file)
# visualize_genome(test_genome)
snn = SpikingNeuralNetwork.create(test_genome, 5.0, **constants.REGULAR_SPIKING_PARAMS)
for sample_food in food:
    eat_counter = 0
    avoid_counter = 0
    input_frequency = high_frequency if sample_food else low_frequency
    reward_frequency = random.choice((high_frequency, low_frequency))
    for time_step in range(20000):
        reward_frequency = get_reward_frequency(sample_food, eat_counter, avoid_counter, reward_frequency)
        inputs = [40 if time_step % input_frequency == 0 else 0,
                  40 if time_step % reward_frequency == 0 else 0]
        snn.set_inputs(inputs)
        outputs = snn.advance(TIME_STEP_IN_MSEC)
        eat_counter += outputs[0]
        avoid_counter += outputs[1]
    print(f'Food: {sample_food}, Eat: {eat_counter}, Avoid: {avoid_counter}')
