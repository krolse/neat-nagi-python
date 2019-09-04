TIME_STEP_IN_MSEC = 0.05
MEMBRANE_POTENTIAL_THRESHOLD = 30.0

REGULAR_SPIKING_PARAMS = {'a': 0.02, 'b': 0.20, 'c': -65.0, 'd': 8.00}
INTRINSICALLY_BURSTING_PARAMS = {'a': 0.02, 'b': 0.20, 'c': -55.0, 'd': 4.00}
CHATTERING_PARAMS = {'a': 0.02, 'b': 0.20, 'c': -50.0, 'd': 2.00}
FAST_SPIKING_PARAMS = {'a': 0.10, 'b': 0.20, 'c': -65.0, 'd': 2.00}
THALAMO_CORTICAL_PARAMS = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 0.05}
RESONATOR_PARAMS = {'a': 0.10, 'b': 0.25, 'c': -65.0, 'd': 2.00}
LOW_THRESHOLD_SPIKING_PARAMS = {'a': 0.02, 'b': 0.25, 'c': -65.0, 'd': 2.00}


class SpikingNeuron(object):
    """Class representing a single spiking neuron."""

    def __init__(self, bias, a, b, c, d, inputs):
        """
        a, b, c, and d are the parameters of the Izhikevich model.

        :param float bias: The bias of the neuron.
        :param float a: The time-scale of the recovery variable.
        :param float b: The sensitivity of the recovery variable.
        :param float c: The after-spike reset value of the membrane potential.
        :param float d: The after-spike reset value of the recovery variable.
        :param list(tuple(int, float)) inputs: A list of (input key, weight) pairs for incoming connections.
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

    def advance(self, dt):
        """
        Advances simulation time by the given time step in milliseconds.

        Update of membrane potential "v" and membrane recovery "u" given by formulas:
            v += dt * (0.04 * v^2 + 5v + 140 - u + I)
            u += dt * a * (b * v - u)

        Once membrane potential exceeds threshold:
            v = c
            u = u + d

        :param float dt: Time step in miliseconds
        :return: void
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
        """
        Resets all state variables.

        :return: void
        """

        self.membrane_potential = self.c
        self.membrane_recovery = self.b * self.membrane_potential

        self.fired = 0
        self.current = self.bias


class SpikingNeuralNetwork(object):
    """Class representing a spiking neural network."""

    def __init__(self, neurons, inputs, outputs):
        """
        :param dict(int, SpikingNeuron) neurons: Dictionary containing key/node pairs.
        :param list(int) inputs: List of input node keys.
        :param list(int) outputs: List of output node keys.
        :var dict(int, float) self.input_values: Dictionary containing input key/voltage pairs.
        """

        self.neurons = neurons
        self.inputs = inputs
        self.outputs = outputs
        self.input_values = {}

