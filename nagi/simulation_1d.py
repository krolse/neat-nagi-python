import random
import re
from enum import Enum
from typing import List, Tuple, Dict

import numpy as np

from nagi.constants import TIME_STEP_IN_MSEC, MAX_HEALTH_POINTS_1D, FLIP_POINT_1D, \
    ACTUATOR_WINDOW, LIF_SPIKE_VOLTAGE, NUM_TIME_STEPS, DAMAGE_FROM_CORRECT_ACTION, \
    DAMAGE_FROM_INCORRECT_ACTION, FOOD_SAMPLES_PER_SIMULATION, DAMAGE_PENALTY_FOR_HIDDEN_NEURONS, PRINT_GREEN, PRINT_RED
from nagi.lifsnn import LIFSpikingNeuralNetwork
from nagi.neat import Genome


class Food(Enum):
    BLACK = 1
    WHITE = 2


class BeneficialFood(Enum):
    BLACK = {Food.BLACK}
    WHITE = {Food.WHITE}
    BOTH = {Food.BLACK, Food.WHITE}
    NONE = {}


class Action(Enum):
    EAT = 1
    AVOID = 2


class OneDimensionalAgent(object):
    def __init__(self, key: int, spiking_neural_network: LIFSpikingNeuralNetwork):
        self.spiking_neural_network = spiking_neural_network
        self.key = key
        self.eat_actuator = 0
        self.avoid_actuator = 0
        self.health_points = MAX_HEALTH_POINTS_1D
        self.current_action = None

    def select_action(self):
        if self.eat_actuator > self.avoid_actuator:
            self.current_action = Action.EAT
        elif self.avoid_actuator > self.eat_actuator:
            self.current_action = Action.AVOID
        return self.current_action

    def reset_actuators(self):
        self.eat_actuator = 0
        self.avoid_actuator = 0

    @staticmethod
    def create_agent(genome: Genome):
        return OneDimensionalAgent(genome.key, LIFSpikingNeuralNetwork.create(genome))


class OneDimensionalEnvironment(object):
    def __init__(self, high_frequency: int, low_frequency: int):
        self.high_frequency = OneDimensionalEnvironment._generate_spike_frequency(high_frequency)
        self.low_frequency = OneDimensionalEnvironment._generate_spike_frequency(low_frequency)
        self.beneficial_food, self.mutator = self._initialize_beneficial_food_and_mutator()
        self.food_loadout = OneDimensionalEnvironment._initialize_food_loadout()
        self.maximum_possible_lifetime = int((len(self.food_loadout) * NUM_TIME_STEPS) / DAMAGE_FROM_CORRECT_ACTION)
        self.minimum_lifetime = int((len(self.food_loadout) * NUM_TIME_STEPS) / DAMAGE_FROM_INCORRECT_ACTION)

    def mutate(self):
        self.beneficial_food = self.mutator[self.beneficial_food]

    def deal_damage(self, agent: OneDimensionalAgent, sample: Food):
        correct_partition, incorrect_partition = self._get_damage_partitions(agent, sample)
        agent.health_points -= (correct_partition * DAMAGE_FROM_CORRECT_ACTION +
                                incorrect_partition * DAMAGE_FROM_INCORRECT_ACTION) ** DAMAGE_PENALTY_FOR_HIDDEN_NEURONS

    def simulate(self, agent: OneDimensionalAgent) -> Tuple[int, float]:
        eat_actuator = []
        avoid_actuator = []
        inputs = self._get_initial_input_voltages()
        for i, sample in enumerate(self.food_loadout):
            eat_actuator = [t for t in eat_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            avoid_actuator = [t for t in avoid_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            frequencies = self._get_initial_input_frequencies(sample)
            if i >= FLIP_POINT_1D and i % FLIP_POINT_1D == 0:
                self.mutate()
            for time_step in range(i * NUM_TIME_STEPS, (i + 1) * NUM_TIME_STEPS):
                if agent.health_points <= 0:
                    return agent.key, self._fitness(time_step)
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
                agent.eat_actuator = OneDimensionalEnvironment._count_spikes_within_time_window(time_step, eat_actuator)
                agent.avoid_actuator = OneDimensionalEnvironment._count_spikes_within_time_window(time_step,
                                                                                                  avoid_actuator)
                self.deal_damage(agent, sample)

        return agent.key, self._fitness(self.maximum_possible_lifetime)

    def simulate_with_visualization(self, agent: OneDimensionalAgent) \
            -> Tuple[
                int, float, dict, dict, int, List[Tuple[int, int]], List[Tuple[int, int]], float, float, List[Food],
                List[BeneficialFood]]:
        eat_actuator = []
        avoid_actuator = []
        weights = {key: [] for key, _ in agent.spiking_neural_network.get_weights().items()}
        membrane_potentials = {key: [] for key, _ in
                               agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items()}
        action_logger = []
        end_of_sample_action_logger = []
        actuator_logger = []
        sample_logger = []
        beneficial_food_logger = []

        inputs = self._get_initial_input_voltages()
        for i, sample in enumerate(self.food_loadout):
            eat_actuator = [t for t in eat_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            avoid_actuator = [t for t in avoid_actuator if t >= NUM_TIME_STEPS * (i - 1)]

            frequencies = self._get_initial_input_frequencies(sample)
            if i >= FLIP_POINT_1D and i % FLIP_POINT_1D == 0:
                print(10 * "=")
                self.mutate()
            sample_logger.append(sample)
            beneficial_food_logger.append(self.beneficial_food)
            for time_step in range(i * NUM_TIME_STEPS, (i + 1) * NUM_TIME_STEPS):
                actuator_logger.append((agent.eat_actuator, agent.avoid_actuator))
                action_logger.append(self._get_correct_wrong_int(agent, sample))
                for key, weight in agent.spiking_neural_network.get_weights().items():
                    weights[key].append(weight)
                for key, membrane_potential in agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items():
                    membrane_potentials[key].append(membrane_potential)

                if agent.health_points <= 0:
                    return (agent.key,
                            self._fitness(time_step),
                            weights,
                            membrane_potentials,
                            time_step,
                            self._get_wrong_action_intervals(action_logger),
                            actuator_logger,
                            sum(action_logger) / len(action_logger),
                            sum(end_of_sample_action_logger) / len(end_of_sample_action_logger),
                            sample_logger,
                            beneficial_food_logger)

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
                agent.eat_actuator = OneDimensionalEnvironment._count_spikes_within_time_window(time_step, eat_actuator)
                agent.avoid_actuator = OneDimensionalEnvironment._count_spikes_within_time_window(time_step,
                                                                                                  avoid_actuator)
                self.deal_damage(agent, sample)
            str_correct_wrong = self._get_correct_wrong_string(agent, sample)
            end_of_sample_action_logger.append(self._get_correct_wrong_int(agent, sample))
            print(
                f'Agent health: {int(agent.health_points)}, i={i}, beneficial food: {self.beneficial_food}, sample: {sample}, action: {agent.select_action()} {str_correct_wrong}')
            print(f'Eat: {agent.eat_actuator}, Avoid: {agent.avoid_actuator}')
        return (agent.key,
                self._fitness(self.maximum_possible_lifetime),
                weights,
                membrane_potentials,
                self.maximum_possible_lifetime,
                self._get_wrong_action_intervals(action_logger),
                actuator_logger,
                sum(action_logger) / len(action_logger),
                sum(end_of_sample_action_logger) / len(end_of_sample_action_logger),
                sample_logger,
                beneficial_food_logger)

    @staticmethod
    def _initialize_food_loadout():
        return [*random.sample([color for color in Food], 2) * int(FOOD_SAMPLES_PER_SIMULATION / Food.__len__())]

    @staticmethod
    def _initialize_beneficial_food_and_mutator() -> Tuple[BeneficialFood, Dict[BeneficialFood, BeneficialFood]]:
        beneficial_foods = [food for food in BeneficialFood]
        ordered_foods = random.sample(beneficial_foods, len(beneficial_foods))
        mutator = {}
        for i in range(0, len(ordered_foods) - 1):
            mutator[ordered_foods[i]] = ordered_foods[i + 1]
        mutator[ordered_foods[-1]] = ordered_foods[0]

        return ordered_foods[0], mutator

    def _get_input_frequencies(self, time_step: int, sample: Food, eat_actuator: List[int], avoid_actuator: List[int],
                               previous_reward_frequencies: List[int]) -> List[int]:
        eat_count = OneDimensionalEnvironment._count_spikes_within_time_window(time_step, eat_actuator)
        avoid_count = OneDimensionalEnvironment._count_spikes_within_time_window(time_step, avoid_actuator)

        return [*self._encode_sample(sample),
                *self._encode_reward(sample, eat_count, avoid_count, previous_reward_frequencies)]

    def _get_initial_input_frequencies(self, sample: Food):
        return [*self._encode_sample(sample), self.low_frequency, self.low_frequency]

    def _encode_sample(self, sample: Food) -> Tuple[int, int]:
        """
        Encodes a food sample into input frequencies.
        """
        return {
            Food.BLACK: (self.high_frequency, self.low_frequency),
            Food.WHITE: (self.low_frequency, self.high_frequency)
        }[sample]

    def _encode_reward(self, sample: Food, eat_count: int, avoid_count: int, previous_reward_frequencies):
        if eat_count > avoid_count:
            if sample in self.beneficial_food.value:
                return self.high_frequency, self.low_frequency
            else:
                return self.low_frequency, self.high_frequency
        elif avoid_count > eat_count:
            if sample not in self.beneficial_food.value:
                return self.high_frequency, self.low_frequency
            else:
                return self.low_frequency, self.high_frequency
        else:
            return previous_reward_frequencies

    def _fitness(self, lifetime: int) -> float:
        """
        Fitness function based on lifetime of an agent during simulation.

        :param lifetime: The lifetime of the agent.
        :return: The fitness score. Ranges between [0, 1].
        """
        return (lifetime - self.minimum_lifetime) / (self.maximum_possible_lifetime - self.minimum_lifetime)

    def _get_damage_partitions(self, agent: OneDimensionalAgent, sample: Food) -> Tuple[float, float]:
        """
        Utility function for determining how much damage to be dealt to an agent based on its actuators.

        :param agent: The agent to be damaged.
        :param sample: A food sample.
        :return: Partitions for correct and incorrect damage. Sums to 1.
        """

        if agent.select_action() is None:
            return 0, 1.0

        if sample in self.beneficial_food.value:
            spikes_correct_action = agent.eat_actuator
            spikes_incorrect_action = agent.avoid_actuator
        else:
            spikes_correct_action = agent.avoid_actuator
            spikes_incorrect_action = agent.eat_actuator

        total_spikes = agent.eat_actuator + agent.avoid_actuator

        if total_spikes == 0:
            correct_partition = 0.5
        elif 0 < total_spikes <= 6:
            norm_spikes_correct_action = np.clip(spikes_correct_action, 0, 3) / 6
            norm_spikes_incorrect_action = np.clip(spikes_incorrect_action, 0, 3) / 6

            correct_partition = norm_spikes_correct_action + (0.5 - norm_spikes_incorrect_action)
        else:
            correct_partition = spikes_correct_action / total_spikes

        return correct_partition, 1 - correct_partition

    @staticmethod
    def _get_input_voltages(time_step: int, frequencies: List[int]):
        return [LIF_SPIKE_VOLTAGE if time_step > frequency and time_step % frequency == 0 else 0 for frequency in
                frequencies]

    @staticmethod
    def _get_initial_input_voltages():
        return [LIF_SPIKE_VOLTAGE, LIF_SPIKE_VOLTAGE, 0, 0]

    @staticmethod
    def _count_spikes_within_time_window(time_step: int, actuator: List[int]):
        return len([t for t in actuator if time_step - t <= ACTUATOR_WINDOW])

    @staticmethod
    def _generate_spike_frequency(frequency: int) -> int:
        return int(1 / (TIME_STEP_IN_MSEC / 1000) / frequency)

    @staticmethod
    def _get_wrong_action_intervals(values: List[int]):
        return [(m.start(), m.end()) for m in re.finditer(r'0+', ''.join([str(x) for x in values]))]

    def _get_correct_wrong_string(self, agent: OneDimensionalAgent, sample: Food) -> str:
        def green(s):
            return f'{PRINT_GREEN}{s}\033[m'

        def red(s):
            return f'{PRINT_RED}{s}\033[m'

        return green("CORRECT") if (agent.select_action() is Action.EAT and sample in self.beneficial_food.value) or (
                agent.select_action() is Action.AVOID and sample not in self.beneficial_food.value) else red("WRONG")

    def _get_correct_wrong_int(self, agent: OneDimensionalAgent, sample: Food) -> int:
        return 1 if (agent.select_action() is Action.EAT and sample in self.beneficial_food.value) or (
                agent.select_action() is Action.AVOID and sample not in self.beneficial_food.value) else 0
