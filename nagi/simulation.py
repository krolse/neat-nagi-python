import pickle
import sys
from enum import Enum
from random import random
from typing import List

from nagi.constants import TIME_STEP_IN_MSEC, MAX_HEALTH_POINTS, DAMAGE_FROM_EATING_CORRECT_FOOD, \
    DAMAGE_FROM_AVOIDING_FOOD, FLIP_POINT, REGULAR_SPIKING_PARAMS, DAMAGE_FROM_EATING_WRONG_FOOD, ACTUATOR_WINDOW, \
    INPUT_SPIKE_VOLTAGE
from nagi.neat import Genome
from nagi.snn import SpikingNeuralNetwork


class Food(Enum):
    WHITE = 1
    BLACK = 2


class Action(Enum):
    EAT = 1
    AVOID = 2


class Agent(object):
    def __init__(self, key: int, spiking_neural_network: SpikingNeuralNetwork):
        self.spiking_neural_network = spiking_neural_network
        self.key = key
        self.eat_actuator = 0
        self.avoid_actuator = 0
        self.health_points = MAX_HEALTH_POINTS

    def select_action(self):
        return Action.EAT if self.eat_actuator > self.avoid_actuator else Action.AVOID

    def reset_actuators(self):
        self.eat_actuator = 0
        self.avoid_actuator = 0

    @staticmethod
    def create_agent(genome: Genome):
        return Agent(genome.key, SpikingNeuralNetwork.create(genome, 5, **REGULAR_SPIKING_PARAMS))


class Environment(object):
    def __init__(self, high_frequency: int, low_frequency: int):
        self.high_frequency = Environment._generate_spike_frequency(high_frequency)
        self.low_frequency = Environment._generate_spike_frequency(low_frequency)
        self.beneficial_food = random.choice([value for value in Food])
        self.food_loadout = self._initialize_food_loadout()
        self.maximum_possible_lifetime = len(self.food_loadout)

    def mutate(self):
        self.beneficial_food = Food.WHITE if self.beneficial_food is Food.BLACK else Food.BLACK

    def deal_damage(self, agent: Agent, sample: Food):
        action = agent.select_action()
        if action is Action.EAT:
            if sample is self.beneficial_food:
                agent.health_points -= DAMAGE_FROM_EATING_CORRECT_FOOD
            else:
                agent.health_points -= DAMAGE_FROM_EATING_WRONG_FOOD
        elif action is Action.AVOID:
            agent.health_points -= DAMAGE_FROM_AVOIDING_FOOD
        agent.reset_actuators()

    def simulate(self, agent: Agent) -> float:
        for i, sample in enumerate(self.food_loadout):
            eat_actuator = []
            avoid_actuator = []
            if agent.health_points <= 0:
                return self._fitness(i)
            frequencies = self._get_initial_input_frequencies(sample)
            inputs = self._get_initial_input_voltages()
            for time_step in range(25000):
                if time_step > 0:
                    if time_step % FLIP_POINT == 0:
                        self.mutate()
                    frequencies = self._get_input_frequencies(time_step, sample, eat_actuator, avoid_actuator,
                                                              frequencies[2:])
                    inputs = self._get_input_voltages(time_step, frequencies)
                agent.spiking_neural_network.set_inputs(inputs)
                eat, avoid = agent.spiking_neural_network.advance(TIME_STEP_IN_MSEC)
                if eat:
                    eat_actuator.append(time_step)
                if avoid:
                    avoid_actuator.append(time_step)
            agent.eat_actuator = Environment._count_spikes_within_time_window(25000 - 1, eat_actuator)
            agent.avoid_actuator = Environment._count_spikes_within_time_window(25000 - 1, avoid_actuator)
            self.deal_damage(agent, sample)

    def _initialize_food_loadout(self):
        initial_loadout = random.choice([value for value in Food], MAX_HEALTH_POINTS)
        beneficial_food = self.beneficial_food
        mock_health = MAX_HEALTH_POINTS
        for i, sample in enumerate(initial_loadout):
            if i > 0 and i % FLIP_POINT == 0:
                beneficial_food = Food.WHITE if beneficial_food == Food.BLACK else Food.BLACK
            mock_health -= DAMAGE_FROM_EATING_CORRECT_FOOD if sample is beneficial_food else DAMAGE_FROM_AVOIDING_FOOD
            if mock_health <= 0:
                return initial_loadout[:i]

    def _get_input_frequencies(self, time_step: int, sample: Food, eat_actuator: List[int], avoid_actuator: List[int],
                               previous_reward_frequencies: List[int]) -> List[int]:
        eat_count = Environment._count_spikes_within_time_window(time_step, eat_actuator)
        avoid_count = Environment._count_spikes_within_time_window(time_step, avoid_actuator)
        (input_frequency_1, input_frequency_2) = (
            self.high_frequency, self.low_frequency) if sample is self.beneficial_food else (
            self.low_frequency, self.high_frequency)
        (reward_frequency_1, reward_frequency_2) = (
            self.high_frequency, self.low_frequency) if eat_count > avoid_count else (
            self.low_frequency, self.high_frequency) if avoid_count > eat_count else previous_reward_frequencies
        return [input_frequency_1, input_frequency_2, reward_frequency_1, reward_frequency_2]

    def _get_initial_input_frequencies(self, sample: Food):
        (input_frequency_1, input_frequency_2) = (
            self.high_frequency, self.low_frequency) if sample is self.beneficial_food else (
            self.low_frequency, self.high_frequency)
        return [input_frequency_1, input_frequency_2, sys.maxsize, sys.maxsize]

    def _fitness(self, lifetime: int):
        return lifetime / self.maximum_possible_lifetime

    @staticmethod
    def _get_input_voltages(time_step: int, frequencies: List[int]):
        return [INPUT_SPIKE_VOLTAGE if time_step % frequency == 0 else 0 for frequency in
                frequencies]

    @staticmethod
    def _get_initial_input_voltages():
        return [INPUT_SPIKE_VOLTAGE, INPUT_SPIKE_VOLTAGE, 0, 0]

    @staticmethod
    def _count_spikes_within_time_window(time_step: int, actuator: List[int]):
        return len([t for t in actuator if time_step - t <= ACTUATOR_WINDOW])

    @staticmethod
    def _generate_spike_frequency(frequency: int) -> int:
        return int(1 / (TIME_STEP_IN_MSEC / 1000) / frequency)

# def generate_spike_train(frequency: int, number_of_seconds: int, time_step_in_msec: float) -> List[int]:
#     number_of_steps = int(number_of_seconds / (time_step_in_msec / 1000))
#     nth = int(number_of_steps / frequency)
#     return [40 if i % nth == 0 else 0 for i in range(number_of_steps)]
#
#
# def generate_spike_frequency(frequency: int, time_step_in_msec: float) -> int:
#     return int(1 / (time_step_in_msec / 1000) / frequency)
#
#
# def get_reward_frequency(sample: int, eat: int, avoid: int, current_frequency: int):
#     if sample:
#         if eat > avoid:
#             return high_frequency
#         elif eat < avoid:
#             return low_frequency
#         else:
#             return current_frequency
#     else:
#         if eat > avoid:
#             return low_frequency
#         elif eat < avoid:
#             return high_frequency
#         else:
#             return current_frequency
#
#
# def swap_frequency(frequency_1):
#     return high_frequency if frequency_1 == low_frequency else low_frequency
#
#
# def sum_of_spikes(counter):
#     return sum([value for key, value in counter.items() if ttime_step - key < max_counter_length])
#
#
# high_frequency = generate_spike_frequency(500, TIME_STEP_IN_MSEC)
# low_frequency = generate_spike_frequency(5, TIME_STEP_IN_MSEC)
# spike_actuator_window = 0.5
# max_counter_length = int(spike_actuator_window / (TIME_STEP_IN_MSEC / 1000))
# ffood = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
#
# with open('../data/test_genome.pkl', 'rb') as file:
#     test_genome = pickle.load(file)
# # visualize_genome(test_genome)
# snn = SpikingNeuralNetwork.create(test_genome, 5.0, **constants.REGULAR_SPIKING_PARAMS)
# for sample_food in ffood:
#     eat_counter = {}
#     avoid_counter = {}
#     iinput_frequency_1 = high_frequency if sample_food else low_frequency
#     iinput_frequency_2 = low_frequency if sample_food else high_frequency
#     rreward_frequency_1 = sys.maxsize
#     rreward_frequency_2 = sys.maxsize
#     for ttime_step in range(25000):
#         rreward_frequency_1 = get_reward_frequency(sample_food, sum_of_spikes(eat_counter),
#                                                    sum_of_spikes(avoid_counter),
#                                                    rreward_frequency_1)
#         rreward_frequency_2 = low_frequency if rreward_frequency_1 == high_frequency else high_frequency
#         iinputs = [1 if ttime_step % iinput_frequency_1 == 0 else 0,
#                    1 if ttime_step % iinput_frequency_2 == 0 else 0,
#                    1 if ttime_step % rreward_frequency_1 == 0 else 0,
#                    1 if ttime_step % rreward_frequency_2 == 0 else 0]
#         snn.set_inputs(iinputs)
#         eat, avoid = snn.advance(TIME_STEP_IN_MSEC)
#         if eat:
#             eat_counter[ttime_step] = 1
#         if avoid:
#             avoid_counter[ttime_step] = 1
#     print(f'Food: {sample_food}, Eat: {sum_of_spikes(eat_counter)}, Avoid: {sum_of_spikes(avoid_counter)}')
