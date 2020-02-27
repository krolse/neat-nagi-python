from enum import Enum
import random
from typing import List, Tuple

from nagi.constants import TIME_STEP_IN_MSEC, MAX_HEALTH_POINTS, DAMAGE_FROM_EATING_CORRECT_FOOD, \
    DAMAGE_FROM_AVOIDING_FOOD, FLIP_POINT, REGULAR_SPIKING_PARAMS, DAMAGE_FROM_EATING_WRONG_FOOD, ACTUATOR_WINDOW, \
    INPUT_SPIKE_VOLTAGE, NUM_TIME_STEPS
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
        self.maximum_possible_lifetime = len(self.food_loadout) - 1
        self.minimum_lifetime = self._get_minimum_lifetime()

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

    def simulate(self, agent: Agent) -> Tuple[int, float]:
        for i, sample in enumerate(self.food_loadout):
            if i >= FLIP_POINT and i % FLIP_POINT == 0:
                self.mutate()
            eat_actuator = []
            avoid_actuator = []
            frequencies = self._get_initial_input_frequencies(sample)
            inputs = self._get_initial_input_voltages()
            for time_step in range(NUM_TIME_STEPS):
                if time_step > 0:
                    frequencies = self._get_input_frequencies(time_step, sample, eat_actuator, avoid_actuator,
                                                              frequencies[2:])
                    inputs = self._get_input_voltages(time_step, frequencies)
                agent.spiking_neural_network.set_inputs(inputs)
                eat, avoid = agent.spiking_neural_network.advance(TIME_STEP_IN_MSEC)
                if eat:
                    eat_actuator.append(time_step)
                if avoid:
                    avoid_actuator.append(time_step)
            agent.eat_actuator = Environment._count_spikes_within_time_window(NUM_TIME_STEPS - 1, eat_actuator)
            agent.avoid_actuator = Environment._count_spikes_within_time_window(NUM_TIME_STEPS - 1, avoid_actuator)
            # print(f'Eat: {agent.eat_actuator}, Avoid: {agent.avoid_actuator}')
            # print(f'Agent health: {agent.health_points}, i={i}, beneficial food: {self.beneficial_food}, sample: {sample}, action: {agent.select_action()}')
            self.deal_damage(agent, sample)
            if agent.health_points <= 0:
                return agent.key, self._fitness(i)

    def _initialize_food_loadout(self):
        initial_loadout = random.choices([value for value in Food], k=MAX_HEALTH_POINTS)
        beneficial_food = self.beneficial_food
        mock_health = MAX_HEALTH_POINTS
        for i, sample in enumerate(initial_loadout):
            if i >= FLIP_POINT and i % FLIP_POINT == 0:
                beneficial_food = Food.WHITE if beneficial_food == Food.BLACK else Food.BLACK
            mock_health -= DAMAGE_FROM_EATING_CORRECT_FOOD if sample is beneficial_food else DAMAGE_FROM_AVOIDING_FOOD
            if mock_health <= 0:
                return initial_loadout[:i + 1]

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
        return [input_frequency_1, input_frequency_2, self.low_frequency, self.low_frequency]

    def _fitness(self, lifetime: int):
        return (lifetime - self.minimum_lifetime) / (self.maximum_possible_lifetime - self.minimum_lifetime)

    def _get_minimum_lifetime(self):
        beneficial_food = self.beneficial_food
        mock_health = MAX_HEALTH_POINTS
        for i, food in enumerate(self.food_loadout):
            if i >= FLIP_POINT and i % FLIP_POINT == 0:
                beneficial_food = Food.WHITE if beneficial_food == Food.BLACK else Food.BLACK
            if food is beneficial_food:
                mock_health -= DAMAGE_FROM_AVOIDING_FOOD
            else:
                mock_health -= DAMAGE_FROM_EATING_WRONG_FOOD
            if mock_health <= 0:
                return i

    @staticmethod
    def _get_input_voltages(time_step: int, frequencies: List[int]):
        return [INPUT_SPIKE_VOLTAGE if time_step > frequency and time_step % frequency == 0 else 0 for frequency in
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
