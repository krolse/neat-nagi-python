from math import exp, sqrt, pi

from nagi.constants import ASYMMETRIC_HEBBIAN_PARAMS, SYMMETRIC_HEBBIAN_PARAMS
from nagi.neat import LearningRule


def asymmetric_hebbian(delta_t: float, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float) -> float:
    """
    Exponential Synaptic Weight Modification function used in STDP based learning.

    :param delta_t: Difference in relative timing of pre- and postsynaptic spikes, in milliseconds.
    :param a_plus: Constant when delta_t > 0.
    :param a_minus: Constant when delta_t < 0.
    :param tau_plus: Time constant when delta_t > 0, in milliseconds.
    :param tau_minus: Time constant when delta_t < 0, in milliseconds.
    :return: Weight modification in decimal percentage.
    """

    if delta_t > 0:
        return a_plus * exp(-delta_t / tau_plus)
    elif delta_t < 0:
        return -a_minus * exp(delta_t / tau_minus)
    else:
        return 0


def asymmetric_anti_hebbian(delta_t: float, a_plus: float, a_minus: float, tau_plus: float, tau_minus: float) -> float:
    """
    Exponential Synaptic Weight Modification function used in STDP based learning.

    :param delta_t: Difference in relative timing of pre- and postsynaptic spikes, in milliseconds.
    :param a_plus: Constant when delta_t > 0.
    :param a_minus: Constant when delta_t < 0.
    :param tau_plus: Time constant when delta_t > 0, in milliseconds.
    :param tau_minus: Time constant when delta_t < 0, in milliseconds.
    :return: Weight modification in decimal percentage.
    """

    return -asymmetric_hebbian(delta_t, a_plus, a_minus, tau_plus, tau_minus)


def symmetric_hebbian(delta_t: float, a_plus: float, a_minus: float, std_plus: float, std_minus: float):
    def gaussian(std: float):
        return exp(-0.5 * (delta_t / std) ** 2) / (std * sqrt(2 * pi))

    difference_of_gaussian = gaussian(std_plus) - gaussian(std_minus)
    a = a_plus if difference_of_gaussian > 0 else a_minus
    return a * difference_of_gaussian


def symmetric_anti_hebbian(delta_t: float, a_plus: float, a_minus: float, std_plus: float, std_minus: float):
    return -symmetric_hebbian(delta_t, a_plus, a_minus, std_plus, std_minus)


def get_learning_rule_function(learning_rule: LearningRule):
    return {
        LearningRule.asymmetric_hebbian: asymmetric_hebbian,
        LearningRule.asymmetric_anti_hebbian: asymmetric_anti_hebbian,
        LearningRule.symmetric_hebbian: symmetric_hebbian,
        LearningRule.symmetric_anti_hebbian: symmetric_anti_hebbian
    }[learning_rule]


def get_learning_rule_params(learning_rule: LearningRule):
    return {
        LearningRule.asymmetric_hebbian: ASYMMETRIC_HEBBIAN_PARAMS,
        LearningRule.asymmetric_anti_hebbian: ASYMMETRIC_HEBBIAN_PARAMS,
        LearningRule.symmetric_hebbian: SYMMETRIC_HEBBIAN_PARAMS,
        LearningRule.symmetric_anti_hebbian: SYMMETRIC_HEBBIAN_PARAMS
    }[learning_rule]
