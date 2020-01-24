import math
from math import exp

from nagi.constants import ASYMMETRIC_HEBBIAN_PARAMS, SYMMETRIC_HEBBIAN_PARAMS
from nagi.neat import LearningRule


def asymmetric_hebbian(delta_t: float, a_plus: float, a_minus: float, b_plus: float, b_minus: float) -> float:
    """
    Exponential Synaptic Weight Modification function used in STDP based learning.

    :param delta_t: Difference in relative timing of pre- and postsynaptic spikes, in milliseconds.
    :param a_plus: Constant when delta_t > 0.
    :param a_minus: Constant when delta_t < 0.
    :param b_plus: Time constant when delta_t > 0, in milliseconds.
    :param b_minus: Time constant when delta_t < 0, in milliseconds.
    :return: Weight modification in decimal percentage.
    """

    if delta_t > 0:
        return -a_plus * exp(-delta_t / b_plus)
    else:
        return a_minus * exp(delta_t / b_minus)


def asymmetric_anti_hebbian(delta_t: float, a_plus: float, a_minus: float, b_plus: float, b_minus: float) -> float:
    """
    Exponential Synaptic Weight Modification function used in STDP based learning.

    :param delta_t: Difference in relative timing of pre- and postsynaptic spikes, in milliseconds.
    :param a_plus: Constant when delta_t > 0.
    :param a_minus: Constant when delta_t < 0.
    :param b_plus: Time constant when delta_t > 0, in milliseconds.
    :param b_minus: Time constant when delta_t < 0, in milliseconds.
    :return: Weight modification in decimal percentage.
    """

    if delta_t > 0:
        return a_plus * exp(-delta_t / b_plus)
    else:
        return -a_minus * exp(delta_t / b_minus)


def symmetric_hebbian(delta_t: float, a_plus: float, a_minus: float, b_plus: float, b_minus: float):
    def gaussian(a: float, std: float):
        return a * math.exp(-0.5 * (delta_t / std) ** 2)

    a_plus, a_minus = max(a_plus, a_minus), min(a_plus, a_minus)
    b_plus, b_minus = min(b_plus, b_minus), max(b_plus, b_minus)

    return gaussian(a_plus, b_plus) - gaussian(a_minus, b_minus)


def symmetric_anti_hebbian(delta_t: float, a_plus: float, a_minus: float, b_plus: float, b_minus: float):
    return -symmetric_hebbian(delta_t, a_plus, a_minus, b_plus, b_minus)


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
