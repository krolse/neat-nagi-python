from enum import Enum


class LearningRule(Enum):
    asymmetric_hebbian = 1
    asymmetric_anti_hebbian = 2
    symmetric_hebbian = 3
    symmetric_anti_hebbian = 4


class NodeGene(object):
    def __init__(self, key: int, is_inhibitory: bool, learning_rule: LearningRule):
        self.key = key
        self.is_inhibitory = is_inhibitory
        self.learning_rule = learning_rule


class ConnectionGene(object):
    def __init__(self, in_node: int, out_node: int):
        self.in_node = in_node
        self.out_node = out_node
        self.is_active = True
