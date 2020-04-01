from enum import Enum
from typing import List, Tuple

from nagi.constants import TIME_STEP_IN_MSEC, MAX_HEALTH_POINTS, FLIP_POINT_1D, \
    ACTUATOR_WINDOW, LIF_SPIKE_VOLTAGE, NUM_TIME_STEPS, DAMAGE_FROM_CORRECT_ACTION, \
    DAMAGE_FROM_INCORRECT_ACTION, FOOD_SAMPLES_PER_SIMULATION, DAMAGE_PENALTY_FOR_HIDDEN_NEURONS
from nagi.lifsnn import LIFSpikingNeuralNetwork
from nagi.neat import Genome


class Food(Enum):
    BLACK = 1
    WHITE = 2


class Action(Enum):
    EAT = 1
    AVOID = 2


class OneDimensionalAgent(object):
    def __init__(self, key: int, spiking_neural_network: LIFSpikingNeuralNetwork):
        self.spiking_neural_network = spiking_neural_network
        self.key = key
        self.eat_actuator = 0
        self.avoid_actuator = 0
        self.health_points = MAX_HEALTH_POINTS
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
        self.food_loadout = OneDimensionalEnvironment._initialize_food_loadout()
        self.maximum_possible_lifetime = int((len(self.food_loadout) * NUM_TIME_STEPS) / DAMAGE_FROM_CORRECT_ACTION)
        self.minimum_lifetime = int((len(self.food_loadout) * NUM_TIME_STEPS) / DAMAGE_FROM_INCORRECT_ACTION)

    @staticmethod
    def mutate(food: Food):
        return {
            Food.BLACK: Food.WHITE,
            Food.WHITE: Food.BLACK
        }[food]

    @staticmethod
    def deal_damage(agent: OneDimensionalAgent, sample: Food, beneficial_food: Food):
        action = agent.select_action()
        if action is Action.EAT:
            if sample is beneficial_food:
                damage = DAMAGE_FROM_CORRECT_ACTION
            else:
                damage = DAMAGE_FROM_INCORRECT_ACTION
        elif action is Action.AVOID:
            if sample is beneficial_food:
                damage = DAMAGE_FROM_INCORRECT_ACTION
            else:
                damage = DAMAGE_FROM_CORRECT_ACTION
        else:
            damage = DAMAGE_FROM_INCORRECT_ACTION
        agent.health_points -= damage * DAMAGE_PENALTY_FOR_HIDDEN_NEURONS ** agent.spiking_neural_network.number_of_hidden_neurons

    def simulate(self, agent: OneDimensionalAgent) -> Tuple[int, float]:
        eat_actuator = []
        avoid_actuator = []
        inputs = self._get_initial_input_voltages()
        beneficial_food = Food.BLACK
        for i, sample in enumerate(self.food_loadout):
            eat_actuator = [t for t in eat_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            avoid_actuator = [t for t in avoid_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            frequencies = self._get_initial_input_frequencies(sample)
            if i >= FLIP_POINT_1D and i % FLIP_POINT_1D == 0:
                # print(10 * "=")
                beneficial_food = self.mutate(sample)
            for time_step in range(i * NUM_TIME_STEPS, (i + 1) * NUM_TIME_STEPS):
                if agent.health_points <= 0:
                    return agent.key, self._fitness(time_step)
                if time_step > 0:
                    frequencies = self._get_input_frequencies(time_step, sample, beneficial_food,
                                                              eat_actuator, avoid_actuator, frequencies[2:])
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
                self.deal_damage(agent, sample, beneficial_food)
            # str_correct_wrong = "CORRECT" if (
            #                         agent.select_action() is Action.EAT and sample is self.beneficial_food) or (
            #                         agent.select_action() is Action.AVOID and sample is not self.beneficial_food) \
            #                     else "WRONG"
            # print(f'Agent health: {int(agent.health_points)}, i={i}, beneficial food: {self.beneficial_food}, sample: {sample}, action: {agent.select_action()} {str_correct_wrong}')
            # print(f'Eat: {agent.eat_actuator}, Avoid: {agent.avoid_actuator}')
        return agent.key, self._fitness(self.maximum_possible_lifetime)

    def simulate_with_visualization(self, agent: OneDimensionalAgent) -> Tuple[int, float, dict, dict, int]:
        eat_actuator = []
        avoid_actuator = []
        beneficial_food = Food.BLACK
        weights = {key: [] for key, _ in agent.spiking_neural_network.get_weights().items()}
        membrane_potentials = {key: [] for key, _ in
                               agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items()}

        inputs = self._get_initial_input_voltages()
        for i, sample in enumerate(self.food_loadout):
            eat_actuator = [t for t in eat_actuator if t >= NUM_TIME_STEPS * (i - 1)]
            avoid_actuator = [t for t in avoid_actuator if t >= NUM_TIME_STEPS * (i - 1)]

            frequencies = self._get_initial_input_frequencies(sample)
            if i >= FLIP_POINT_1D and i % FLIP_POINT_1D == 0:
                print(10 * "=")
                beneficial_food = self.mutate(sample)
            for time_step in range(i * NUM_TIME_STEPS, (i + 1) * NUM_TIME_STEPS):
                for key, weight in agent.spiking_neural_network.get_weights().items():
                    weights[key].append(weight)
                for key, membrane_potential in agent.spiking_neural_network.get_membrane_potentials_and_thresholds().items():
                    membrane_potentials[key].append(membrane_potential)
                if agent.health_points <= 0:
                    return agent.key, self._fitness(time_step), weights, membrane_potentials, time_step
                if time_step > 0:
                    frequencies = self._get_input_frequencies(time_step, sample, beneficial_food,
                                                              eat_actuator, avoid_actuator, frequencies[2:])
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
                self.deal_damage(agent, sample, beneficial_food)
            str_correct_wrong = "CORRECT" if (
                 agent.select_action() is Action.EAT and sample is beneficial_food) or (
                 agent.select_action() is Action.AVOID and sample is not beneficial_food) else "WRONG"
            print(
                f'Agent health: {int(agent.health_points)}, i={i}, beneficial food: {beneficial_food}, sample: {sample}, action: {agent.select_action()} {str_correct_wrong}')
            print(f'Eat: {agent.eat_actuator}, Avoid: {agent.avoid_actuator}')
        return agent.key, self._fitness(
            self.maximum_possible_lifetime), weights, membrane_potentials, self.maximum_possible_lifetime

    @staticmethod
    def _initialize_food_loadout():
        return [*[color for color in Food] * int(FOOD_SAMPLES_PER_SIMULATION / Food.__len__())]

    def _get_input_frequencies(self, time_step: int, sample: Food, beneficial_food: Food, eat_actuator: List[int],
                               avoid_actuator: List[int], previous_reward_frequencies: List[int]) -> List[int]:
        eat_count = OneDimensionalEnvironment._count_spikes_within_time_window(time_step, eat_actuator)
        avoid_count = OneDimensionalEnvironment._count_spikes_within_time_window(time_step, avoid_actuator)

        return [*self._encode_sample(sample),
                *self._encode_reward(sample, eat_count, avoid_count, previous_reward_frequencies, beneficial_food)]

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

    def _encode_reward(self, sample: Food, eat_count: int, avoid_count: int, previous_reward_frequencies,
                       beneficial_food: Food):
        if eat_count > avoid_count:
            if sample is beneficial_food:
                return self.high_frequency, self.low_frequency
            else:
                return self.low_frequency, self.high_frequency
        elif avoid_count > eat_count:
            if sample is not beneficial_food:
                return self.high_frequency, self.low_frequency
            else:
                return self.low_frequency, self.high_frequency
        else:
            return previous_reward_frequencies

    def _fitness(self, lifetime: int):
        return (lifetime - self.minimum_lifetime) / (self.maximum_possible_lifetime - self.minimum_lifetime)

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
