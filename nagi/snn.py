from enum import Enum
from typing import List, Dict

import numpy as np

from nagi.constants import MEMBRANE_POTENTIAL_THRESHOLD, STDP_PARAMS, STDP_LEARNING_WINDOW, NEURON_WEIGHT_BUDGET, \
    THRESHOLD_THETA_INCREMENT_RATE, THRESHOLD_THETA_DECAY_RATE, MAX_THRESHOLD_THETA
from nagi.neat import Genome, NeuralNodeGene, InputNodeGene, OutputNodeGene
from nagi.stdp import *


class StdpType(Enum):
    input = 1
    output = 2


class SpikingNeuron(object):
    """Class representing a single spiking neuron."""

    def __init__(self, bias: float, a: float, b: float, c: float, d: float, inputs: List[int],
                 learning_rule: LearningRule, stdp_parameters: Dict[str, float]):
        """
        a, b, c, and d are the parameters of the Izhikevich model.

        :param bias: The bias of the neuron.
        :param a: The time-scale of the recovery variable.
        :param b: The sensitivity of the recovery variable.
        :param c: The after-spike reset value of the membrane potential.
        :param d: The after-spike reset value of the recovery variable.
        :param inputs: A dictionary of incoming connection weights.
        """

        self.bias = bias
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.inputs = {key: np.random.random() * NEURON_WEIGHT_BUDGET for key in inputs}
        self._normalize_weights()
        self.learning_rule = learning_rule
        self.stdp_parameters = stdp_parameters

        self.membrane_potential = self.c
        self.membrane_recovery = self.b * self.membrane_potential
        self.fired = 0
        self.current = self.bias
        self.threshold_theta = 0

        # Variables containing time elapsed since last input and output spikes.
        self.output_spike_timing: float = 0
        self.input_spike_timings: Dict[int, List[float]] = {key: [] for key in self.inputs.keys()}
        self.has_fired = False

    def advance(self, dt: float):
        """
        Advances simulation time by the given time step in milliseconds.

        Update of membrane potential "v" and membrane recovery "u" given by formulas:
            v += dt * (0.04 * v^2 + 5v + 140 - u + I)
            u += dt * a * (b * v - u)

        Once membrane potential exceeds threshold:
            v = c
            u = u + d

        :param dt: Time step in milliseconds.
        """

        v = self.membrane_potential
        u = self.membrane_recovery

        self.membrane_potential += dt * (0.04 * v ** 2 + 5 * v + 140 - u + self.current)
        self.membrane_recovery += dt * self.a * (self.b * v - u)

        self.fired = 0
        self.output_spike_timing += dt

        for key in self.input_spike_timings.keys():
            # STDP update on received input spike.
            if 0 in self.input_spike_timings[key] and self.has_fired:
                self.stpd_update(key, StdpType.input)

            self.input_spike_timings[key] = [t + dt for t in self.input_spike_timings[key] if
                                             t + dt < STDP_LEARNING_WINDOW]

        if self.membrane_potential > MEMBRANE_POTENTIAL_THRESHOLD + self.threshold_theta:
            self.fired = 1
            self.has_fired = True
            self.membrane_potential = self.c
            self.membrane_recovery += self.d
            self.threshold_theta *= THRESHOLD_THETA_INCREMENT_RATE * ((MAX_THRESHOLD_THETA - self.threshold_theta) /
                                                                      MAX_THRESHOLD_THETA)
            self.output_spike_timing = 0

            # STDP on output spike.
            for key in self.input_spike_timings.keys():
                self.stpd_update(key, StdpType.output)
        else:
            self.threshold_theta *= (1 - THRESHOLD_THETA_DECAY_RATE)

    def reset(self):
        """ Resets all state variables."""

        self.membrane_potential = self.c
        self.membrane_recovery = self.b * self.membrane_potential

        self.fired = 0
        self.current = self.bias

        self.output_spike_timing = 0
        self.input_spike_timings = {key: 0 for key in self.inputs.keys()}

    def apply_learning_rule(self, delta_t: float):
        return get_learning_rule_function(self.learning_rule)(delta_t, **self.stdp_parameters)

    def stpd_update(self, key: int, stdp_type: StdpType):
        """
        Applies STDP to the weight with the supplied key.

        :param stdp_type:
        :param key: The key identifying the synapse weight to be updated.
        :return: void
        """

        delta_weight = 0
        weight = self.inputs[key]
        sigma, w_min, w_max = STDP_PARAMS['sigma'], STDP_PARAMS['w_min'], STDP_PARAMS['w_max']

        if stdp_type is StdpType.input:
            delta_t = self.output_spike_timing - 0
            if abs(delta_t) < STDP_LEARNING_WINDOW:
                delta_weight = self.apply_learning_rule(delta_t)

        elif stdp_type is StdpType.output:
            for input_spike_timing in self.input_spike_timings[key]:
                delta_t = self.output_spike_timing - input_spike_timing
                if abs(delta_t) < STDP_LEARNING_WINDOW:
                    delta_weight += self.apply_learning_rule(delta_t)

        if delta_weight > 0:
            self.inputs[key] += sigma * delta_weight * (w_max - weight)
        elif delta_weight < 0:
            self.inputs[key] += sigma * delta_weight * (weight - abs(w_min))

        self._normalize_weights()

    def _normalize_weights(self):
        sum_of_input_weights = sum(self.inputs.values())
        if sum_of_input_weights > NEURON_WEIGHT_BUDGET:
            self.inputs = {key: value * NEURON_WEIGHT_BUDGET / sum_of_input_weights for key, value in
                           self.inputs.items()}


class SpikingNeuralNetwork(object):
    """Class representing a spiking neural network."""

    def __init__(self, neurons: Dict[int, SpikingNeuron], inputs: List[int], outputs: List[int]):
        """
        :param neurons: Dictionary containing key/node pairs.
        :param inputs: List of input node keys.
        :param outputs: List of output node keys.
        :var self.input_values: Dictionary containing input key/voltage pairs.
        """

        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.input_values: Dict[int, float] = {}

    def set_inputs(self, inputs: List[float]):
        """
        Assigns voltages to the input nodes.

        :param inputs: List of voltage values."""

        assert len(inputs) == len(
            self.inputs), f"Number of inputs {len(inputs)} does not match number of input nodes {len(self.inputs)} "

        for key, voltage in zip(self.inputs, inputs):
            self.input_values[key] = voltage

    def advance(self, dt: float) -> List[float]:
        """
        Advances the neural network with the given input values and neuron states. Iterates through each neuron, then
        through each input of each neuron and evaluates the values to advance the network. The values can come from
        either input nodes, or firing neurons in a previous layer.

        :param dt: Time step in miliseconds.
        :return: List of the output values of the network after advance."""

        for neuron in self.neurons.values():
            neuron.current = neuron.bias
            for key, weight in neuron.inputs.items():
                in_neuron = self.neurons.get(key)
                if in_neuron is not None:
                    in_value = in_neuron.fired
                else:
                    in_value = self.input_values[key]

                # Trigger STDP on received input spike.
                if in_value:
                    neuron.input_spike_timings[key].append(0)

                neuron.current += in_value * weight
                neuron.advance(dt)

        return [self.neurons[key].fired for key in self.outputs]

    def reset(self):
        """Resets all state variables in all neurons in the entire neural network."""
        for neuron in self.neurons.values():
            neuron.reset()

    @staticmethod
    def create(genome: Genome, bias: float, a: float, b: float, c: float, d: float):
        learning_nodes = {key: node for key, node in genome.nodes.items() if isinstance(node, NeuralNodeGene)}
        node_inputs = {key: [] for key in learning_nodes.keys()}
        input_keys = [node.key for node in genome.nodes.values() if isinstance(node, InputNodeGene)]
        output_keys = [node.key for node in genome.nodes.values() if isinstance(node, OutputNodeGene)]

        for connection_gene in genome.get_enabled_connections():
            node_inputs[connection_gene.destination_node].append(connection_gene.origin_node)

        neurons = {key: SpikingNeuron(bias, a, b, c, d, inputs, learning_nodes[key].learning_rule,
                                      learning_nodes[key].stdp_parameters)
                   for key, inputs in node_inputs.items()}

        return SpikingNeuralNetwork(neurons, input_keys, output_keys)
