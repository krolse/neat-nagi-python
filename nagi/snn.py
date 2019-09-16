from math import exp
from typing import List, Dict
from nagi.constants import *


class SpikingNeuron(object):
    """Class representing a single spiking neuron."""

    def __init__(self, bias: float, a: float, b: float, c: float, d: float, inputs: Dict[int, float]):
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
        self.inputs = inputs

        self.membrane_potential = self.c
        self.membrane_recovery = self.b * self.membrane_potential

        self.fired = 0
        self.current = self.bias

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
        if self.membrane_potential > 30.0:
            self.fired = 1
            self.membrane_potential = 0.0
            self.membrane_recovery += self.d

    def reset(self):
        """ Resets all state variables."""

        self.membrane_potential = self.c
        self.membrane_recovery = self.b * self.membrane_potential

        self.fired = 0
        self.current = self.bias

    def stpd_update(self, key: int, dt: float):
        """
        Applies STDP to the weight with the supplied key, dependent on the value of dt.

        :param dt: Difference in the relative timing of pre- and postsynaptic spikes.
        :param key: The key identifying the synapse weight to be updated.
        :return: void
        """

        # noinspection PyShadowingNames
        def exp_synaptic_weight_modification(dt: float, params: Dict[str, float]) -> float:
            """
            Exponential Synaptic Weight Modification function used in STDP based learning.

            :param dt: Difference in the relative timing of pre- and postsynaptic spikes.
            :param params: Dictionary containing hyperparameters for STDP.
            :return: Weight modification in decimal percentage.
            """

            if dt > 0:
                return -params['A+'] * exp(-dt / params['tau+'])
            else:
                return params['A-'] * exp(dt / params['tau-'])

        params = EXPONENTIAL_STDP_PARAMETERS
        weight = self.inputs[key]
        delta_weight = exp_synaptic_weight_modification(dt, params)

        if delta_weight > 0:
            self.inputs[key] += params['sigma'] * delta_weight * (weight - abs(params['w_min']))
        elif delta_weight < 0:
            self.inputs[key] += params['sigma'] * delta_weight * (params['w_max'] - weight)


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

        assert len(inputs) == len(self.inputs), f"Number of inputs {len(inputs)} does not match number of input nodes {len(self.inputs)} "

        for key, voltage in zip(self.inputs, inputs):
            self.input_values[key] = voltage

    def advance(self, dt: float) -> List[float]:
        """
        Advances the neural network with the given input values and neuron states. Iterates through each neuron, then
        through each input of each neuron and evaluates the values to advance the network. The values can come from
        either input nodes, or firing neurons in a previous layer.

        :param float dt: Time step in miliseconds.
        :return: List of the output values of the network after advance."""

        for neuron in self.neurons.values():
            neuron.current = neuron.bias
            for key, weight in neuron.inputs.values():
                in_neuron = self.neurons.get(key)
                if in_neuron is not None:
                    in_value = in_neuron.fired
                else:
                    in_value = self.input_values[key]

                neuron.current += in_value * weight
                neuron.advance(dt)

        return [self.neurons[key].fired for key in self.outputs]

    def reset(self):
        """Resets all state variables in all neurons in the entire neural network."""
        for neuron in self.neurons.values():
            neuron.reset()
