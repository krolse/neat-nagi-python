import random
import re
from enum import Enum
from typing import List, Tuple

import numpy as np

from nagi.constants import TIME_STEP_IN_MSEC, MAX_HEALTH_POINTS_2D, FLIP_POINT_2D, \
    ACTUATOR_WINDOW, LIF_SPIKE_VOLTAGE, NUM_TIME_STEPS, DAMAGE_FROM_CORRECT_ACTION, \
    DAMAGE_FROM_INCORRECT_ACTION, INPUT_SAMPLES_PER_SIMULATION, DAMAGE_PENALTY_FOR_HIDDEN_NEURONS, PRINT_GREEN, \
    PRINT_RED
from nagi.lifsnn import LIFSpikingNeuralNetwork
from nagi.neat import Genome


class LogicGate(Enum):
    """
    Enum values are the sets containing the input cases where the output of the logic gate is 1.
    """

    # Training
    A = {(1, 0), (1, 1)}
    B = {(0, 1), (1, 1)}
    NOT_A = {(0, 0), (0, 1)}
    NOT_B = {(0, 0), (1, 0)}
    CONSTANT_0 = {}
    CONSTANT_1 = {(0, 0), (0, 1), (1, 0), (1, 1)}
    XOR = {(0, 1), (1, 0)}
    XNOR = {(0, 0), (1, 1)}
    # Testing
    AND = {(1, 1)}
    NAND = {(0, 0), (0, 1), (1, 0)}
    OR = {(0, 1), (1, 0), (1, 1)}
    NOR = {(0, 0)}

    @staticmethod
    def get_training_gates():
        return [LogicGate.A, LogicGate.B, LogicGate.NOT_A, LogicGate.NOT_B, LogicGate.CONSTANT_0, LogicGate.CONSTANT_1,
                LogicGate.XOR, LogicGate.XNOR]

    @staticmethod
    def get_testing_gates():
        return [LogicGate.AND, LogicGate.NAND, LogicGate.OR, LogicGate.NOR]


class TwoDimensionalAgent(object):
    def __init__(self, key: int, spiking_neural_network: LIFSpikingNeuralNetwork):
        self.spiking_neural_network = spiking_neural_network
        self.key = key
        self.zero_actuator = 0
        self.one_actuator = 0
        self.health_points = MAX_HEALTH_POINTS_2D
        self.prediction = None

    def select_prediction(self):
        if self.zero_actuator > self.one_actuator:
            self.prediction = 0
        elif self.one_actuator > self.zero_actuator:
            self.prediction = 1
        return self.prediction

    def reset_actuators(self):
        self.zero_actuator = 0
        self.one_actuator = 0

    @staticmethod
    def create_agent(genome: Genome):
        return TwoDimensionalAgent(genome.key, LIFSpikingNeuralNetwork.create(genome))


class TwoDimensionalEnvironment(object):
    def __init__(self, high_frequency: int, low_frequency: int, testing=False):
        self.high_frequency = self._generate_spike_frequency(high_frequency)
        self.low_frequency = self._generate_spike_frequency(low_frequency)
        self.input_loadout = self._initialize_input_loadout()
        self.current_logic_gate, self._mutator = self._initialize_logic_gate_and_mutator(testing)
        self.maximum_possible_lifetime = int((len(self.input_loadout) * NUM_TIME_STEPS) / DAMAGE_FROM_CORRECT_ACTION)
        self.minimum_lifetime = int((len(self.input_loadout) * NUM_TIME_STEPS) / DAMAGE_FROM_INCORRECT_ACTION)

    def mutate(self):
        self.current_logic_gate = self._mutator[self.current_logic_gate]

    def deal_damage(self, agent: TwoDimensionalAgent, sample: Tuple[int, int]):
        correct_partition, incorrect_partition = self._get_damage_partitions(agent, sample)
        agent.health_points -= (correct_partition * DAMAGE_FROM_CORRECT_ACTION +
                                incorrect_partition * DAMAGE_FROM_INCORRECT_ACTION) ** DAMAGE_PENALTY_FOR_HIDDEN_NEURONS

    def simulate(self, agent: TwoDimensionalAgent) -> Tuple[int, float]:
        zero_actuator = []
        one_actuator = []
        inputs = self._get_initial_input_voltages()
        for i, sample in enumerate(self.input_loadout):
            zero_actuator = [t for t in zero_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            one_actuator = [t for t in one_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            frequencies = self._get_initial_input_frequencies(sample)
            if i >= FLIP_POINT_2D and i % FLIP_POINT_2D == 0:
                self.mutate()
            for time_step in range(i * NUM_TIME_STEPS, (i + 1) * NUM_TIME_STEPS):
                if agent.health_points <= 0:
                    return agent.key, self._fitness(time_step)
                if time_step > 0:
                    frequencies = self._get_input_frequencies(time_step, sample, zero_actuator, one_actuator,
                                                              frequencies[4:])
                    inputs = self._get_input_voltages(time_step, frequencies)

                agent.spiking_neural_network.set_inputs(inputs)
                zero, one = agent.spiking_neural_network.advance(TIME_STEP_IN_MSEC)
                if zero:
                    zero_actuator.append(time_step)
                if one:
                    one_actuator.append(time_step)
                agent.zero_actuator = TwoDimensionalEnvironment._count_spikes_within_time_window(time_step,
                                                                                                 zero_actuator)
                agent.one_actuator = TwoDimensionalEnvironment._count_spikes_within_time_window(time_step, one_actuator)
                self.deal_damage(agent, sample)
        return agent.key, self._fitness(self.maximum_possible_lifetime)

    def simulate_with_visualization(self, agent: TwoDimensionalAgent) -> \
            Tuple[int, float, dict, dict, int, List[Tuple[int, int]], List[Tuple[int, int]], float, float]:
        zero_actuator = []
        one_actuator = []
        weights = {key: [] for key, _ in agent.spiking_neural_network.get_weights().items()}
        membrane_potentials = {key: [] for key, _ in
                               agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items()}
        prediction_logger = []
        end_of_sample_prediction_logger = []
        actuator_logger = []
        inputs = self._get_initial_input_voltages()
        for i, sample in enumerate(self.input_loadout):
            zero_actuator = [t for t in zero_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            one_actuator = [t for t in one_actuator if t >= NUM_TIME_STEPS * (i - 1)]

            frequencies = self._get_initial_input_frequencies(sample)
            if i >= FLIP_POINT_2D and i % FLIP_POINT_2D == 0:
                print(10 * "=")
                self.mutate()
            for time_step in range(i * NUM_TIME_STEPS, (i + 1) * NUM_TIME_STEPS):
                actuator_logger.append((agent.zero_actuator, agent.one_actuator))
                prediction_logger.append(self._get_correct_wrong_int(agent, sample))
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
                            self._get_wrong_prediction_intervals(prediction_logger),
                            actuator_logger,
                            sum(prediction_logger) / len(prediction_logger),
                            sum(end_of_sample_prediction_logger) / len(end_of_sample_prediction_logger))
                if time_step > 0:
                    frequencies = self._get_input_frequencies(time_step, sample, zero_actuator, one_actuator,
                                                              frequencies[4:])
                    inputs = self._get_input_voltages(time_step, frequencies)
                agent.spiking_neural_network.set_inputs(inputs)
                zero, one = agent.spiking_neural_network.advance(TIME_STEP_IN_MSEC)
                if zero:
                    zero_actuator.append(time_step)
                if one:
                    one_actuator.append(time_step)
                agent.zero_actuator = TwoDimensionalEnvironment._count_spikes_within_time_window(time_step,
                                                                                                 zero_actuator)
                agent.one_actuator = TwoDimensionalEnvironment._count_spikes_within_time_window(time_step, one_actuator)
                self.deal_damage(agent, sample)
            end_of_sample_prediction_logger.append(self._get_correct_wrong_int(agent, sample))
            str_correct_wrong = self._get_correct_wrong_string(agent, sample)
            print(
                f'Agent health: {int(agent.health_points)}, i={i}, current_logic_gate: {self.current_logic_gate}, sample: {sample}, prediction: {agent.select_prediction()} {str_correct_wrong}')
            print(f'Zero: {agent.zero_actuator}, One: {agent.one_actuator}')
        return (agent.key,
                self._fitness(self.maximum_possible_lifetime),
                weights,
                membrane_potentials,
                self.maximum_possible_lifetime,
                self._get_wrong_prediction_intervals(prediction_logger),
                actuator_logger,
                sum(prediction_logger) / len(prediction_logger),
                sum(end_of_sample_prediction_logger) / len(end_of_sample_prediction_logger))

    @staticmethod
    def _initialize_input_loadout():
        return [*random.sample([(0, 0), (0, 1), (1, 0), (1, 1)], 4) * int(INPUT_SAMPLES_PER_SIMULATION / 4)]

    @staticmethod
    def _initialize_logic_gate_and_mutator(testing: bool):
        if testing:
            gates = LogicGate.get_testing_gates()
        else:
            gates = LogicGate.get_training_gates()
        ordered_gates = random.sample(gates, len(gates))
        mutator = {}
        for i in range(0, len(ordered_gates) - 1):
            mutator[ordered_gates[i]] = ordered_gates[i + 1]
        mutator[ordered_gates[-1]] = ordered_gates[0]

        return ordered_gates[0], mutator

    def _get_input_frequencies(self, time_step: int, sample: Tuple[int, int], zero_actuator: List[int],
                               one_actuator: List[int], previous_reward_frequencies: List[int]) -> List[int]:
        zero_count = TwoDimensionalEnvironment._count_spikes_within_time_window(time_step, zero_actuator)
        one_count = TwoDimensionalEnvironment._count_spikes_within_time_window(time_step, one_actuator)

        return [*self._encode_sample(sample),
                *self._encode_reward(sample, zero_count, one_count, previous_reward_frequencies)]

    def _get_initial_input_frequencies(self, sample: Tuple[int, int]):
        return [*self._encode_sample(sample), self.low_frequency, self.low_frequency]

    def _encode_sample(self, sample: Tuple[int, int]):
        return {
            (0, 0): (self.high_frequency, self.low_frequency, self.high_frequency, self.low_frequency),
            (0, 1): (self.high_frequency, self.low_frequency, self.low_frequency, self.high_frequency),
            (1, 0): (self.low_frequency, self.high_frequency, self.high_frequency, self.low_frequency),
            (1, 1): (self.low_frequency, self.high_frequency, self.low_frequency, self.high_frequency)
        }[sample]

    def _encode_reward(self, sample: Tuple[int, int], zero_count: int, one_count: int, previous_reward_frequencies):
        if one_count > zero_count:
            if sample in self.current_logic_gate.value:
                return self.high_frequency, self.low_frequency
            else:
                return self.low_frequency, self.high_frequency
        elif zero_count > one_count:
            if sample not in self.current_logic_gate.value:
                return self.high_frequency, self.low_frequency
            else:
                return self.low_frequency, self.high_frequency
        else:
            return previous_reward_frequencies

    def _fitness(self, lifetime: int):
        return (lifetime - self.minimum_lifetime) / (self.maximum_possible_lifetime - self.minimum_lifetime)

    def _get_damage_partitions(self, agent: TwoDimensionalAgent, sample: Tuple[int, int]) -> Tuple[float, float]:
        """
        Utility function for determining how much damage to be dealt to an agent based on its actuators.

        :param agent: The agent to be damaged.
        :param sample: A food sample.
        :return: Partitions for correct and incorrect damage. Sums to 1.
        """

        if agent.select_prediction() is None:
            return 0, 1.0

        if sample in self.current_logic_gate.value:
            spikes_correct_prediction = agent.one_actuator
            spikes_incorrect_prediction = agent.zero_actuator
        else:
            spikes_correct_prediction = agent.zero_actuator
            spikes_incorrect_prediction = agent.one_actuator

        total_spikes = agent.zero_actuator + agent.one_actuator

        if total_spikes == 0:
            correct_partition = 0.5
        elif 0 < total_spikes <= 6:
            norm_spikes_correct_prediction = np.clip(spikes_correct_prediction, 0, 3) / 6
            norm_spikes_incorrect_prediction = np.clip(spikes_incorrect_prediction, 0, 3) / 6

            correct_partition = norm_spikes_correct_prediction + (0.5 - norm_spikes_incorrect_prediction)
        else:
            correct_partition = spikes_correct_prediction / total_spikes

        return correct_partition, 1 - correct_partition

    @staticmethod
    def _get_input_voltages(time_step: int, frequencies: List[int]):
        return [LIF_SPIKE_VOLTAGE if time_step > frequency and time_step % frequency == 0 else 0 for frequency in
                frequencies]

    @staticmethod
    def _get_initial_input_voltages():
        return [LIF_SPIKE_VOLTAGE, LIF_SPIKE_VOLTAGE, LIF_SPIKE_VOLTAGE, LIF_SPIKE_VOLTAGE, 0, 0]

    @staticmethod
    def _count_spikes_within_time_window(time_step: int, actuator: List[int]):
        return len([t for t in actuator if time_step - t <= ACTUATOR_WINDOW])

    @staticmethod
    def _generate_spike_frequency(frequency: int) -> int:
        return int(1 / (TIME_STEP_IN_MSEC / 1000) / frequency)

    @staticmethod
    def _get_wrong_prediction_intervals(values: List[int]):
        return [(m.start(), m.end()) for m in re.finditer(r'0+', ''.join([str(x) for x in values]))]

    def _get_correct_wrong_string(self, agent: TwoDimensionalAgent, sample: Tuple[int, int]) -> str:
        def green(s):
            return f'{PRINT_GREEN}{s}\033[m'

        def red(s):
            return f'{PRINT_RED}{s}\033[m'

        return green("CORRECT") if (agent.select_prediction() == 1 and sample in self.current_logic_gate.value) or (
                agent.select_prediction() == 0 and sample not in self.current_logic_gate.value) else red("WRONG")

    def _get_correct_wrong_int(self, agent: TwoDimensionalAgent, sample: Tuple[int, int]) -> int:
        return 1 if (agent.select_prediction() == 1 and sample in self.current_logic_gate.value) or (
                agent.select_prediction() == 0 and sample not in self.current_logic_gate.value) else 0
