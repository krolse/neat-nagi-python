from math import exp


def asymmetric_hebbian(dt: float, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float) -> float:
    """
    Exponential Synaptic Weight Modification function used in STDP based learning.

    :param dt: Difference in relative timing of pre- and postsynaptic spikes, in milliseconds.
    :param a_plus:
    :param a_minus:
    :param tau_plus:
    :param tau_minus:
    :return: Weight modification in decimal percentage.
    """

    if dt > 0:
        return -a_plus * exp(-dt / tau_plus)
    else:
        return a_minus * exp(dt / tau_minus)


def asymmetric_anti_hebbian(dt: float, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float) -> float:
    """
    Exponential Synaptic Weight Modification function used in STDP based learning.

    :param dt: Difference in relative timing of pre- and postsynaptic spikes, in milliseconds.
    :param a_plus:
    :param a_minus:
    :param tau_plus:
    :param tau_minus:
    :return: Weight modification in decimal percentage.
    """

    if dt > 0:
        return a_plus * exp(-dt / tau_plus)
    else:
        return -a_minus * exp(dt / tau_minus)

