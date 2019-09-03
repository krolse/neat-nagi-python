class SpikingNeuron(object):
    def __init__(self, bias, a, b, c, d, inputs):
        self.bias = bias
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.inputs = inputs

        self.membrane_potential = self.c
        self.membrane_recovery = self.b * self.membrane_potential

        self.fired = 0.0
        self.current = self.bias

    def advance(self, dt):
        v = self.membrane_potential
        u = self.membrane_recovery

        self.membrane_potential += 0.5 * dt * (0.04 * v ** 2 + 5 * v + 140 - u + self.current)
        self.membrane_recovery += dt * self.a * (self.b * v - u)

        self.fired = 0.0
        if self.membrane_potential > 30.0:
            self.fired = 1.0
            self.membrane_potential = 0.0
            self.membrane_recovery += self.d

    def reset(self):
        self.membrane_potential = self.c
        self.membrane_recovery = self.b * self.membrane_potential

        self.fired = 0.0
        self.current = self.bias

