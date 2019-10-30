from collections import defaultdict

import numpy as np
import random
from enum import Enum

from nagi.constants import ENABLE_MUTATE_RATE, ADD_CONNECTION_MUTATE_RATE, MAX_CONNECTION_MUTATION_ATTEMPTS, \
    ADD_NODE_MUTATE_RATE


class LearningRule(Enum):
    asymmetric_hebbian = 1
    asymmetric_anti_hebbian = 2
    symmetric_hebbian = 3
    symmetric_anti_hebbian = 4


class NodeType(Enum):
    input = 1
    output = 2
    hidden = 3


class NodeGene(object):
    def __init__(self, key: int, node_type: NodeType):
        self.key = key
        self.node_type = node_type


class HiddenNodeGene(NodeGene):
    # TODO: Should the hidden node gene have a bias, or should the bias be randomly initialized like weights?
    def __init__(self, key: int, node_type: NodeType, is_inhibitory: bool = False,
                 learning_rule: LearningRule = LearningRule.asymmetric_hebbian):
        super(HiddenNodeGene, self).__init__(key, node_type)
        self.is_inhibitory = is_inhibitory
        self.learning_rule = learning_rule


class ConnectionGene(object):
    def __init__(self, in_node: int, out_node: int, innovation_number: int):
        self.in_node = in_node
        self.out_node = out_node
        self.innovation_number = innovation_number
        self.enabled = True

    def mutate(self):
        if np.random.random() < ENABLE_MUTATE_RATE:
            self.enabled = not self.enabled


class Genome(object):
    def __init__(self, key: int, input_size: int, output_size: int):
        self.key = key
        self.input_size = input_size
        self.output_size = output_size

        self.nodes = {}
        self.connections = {}
        self.input_keys = [i for i in range(input_size)]
        self.output_keys = [i for i in range(input_size, input_size + output_size)]

        # Initialize node and connection genes for inputs and outputs.
        # TODO: Innovation numbers need to be kept track of by the NEAT-algorithm itself in a global innovations list.
        # TODO: Should self.nodes and self.connections be dictionairies or lists?

        innovation_number = 0
        for input_key in self.input_keys:
            self.nodes[input_key] = NodeGene(input_key, NodeType.input)
            for output_key in self.output_keys:
                if self.nodes[output_key] is None:
                    self.nodes[output_key] = NodeGene(output_key, NodeType.output)
                self.connections[innovation_number] = ConnectionGene(input_key, output_key, innovation_number)
                innovation_number += 1

    def mutate_add_connection(self):
        if random.random() < ADD_CONNECTION_MUTATE_RATE:
            (origin_node, destination_node) = random.choice(
                [(origin_node.key, destionation_node.key)
                 for origin_node in self.nodes.values()
                 for destionation_node in self.nodes.values()
                 if (origin_node, destionation_node) not in self.connections
                 and origin_node.node_type is not NodeType.output
                 and destionation_node is not NodeType.input])

            innovation_number = len(self.connections)
            self.connections[innovation_number] = ConnectionGene(origin_node, destination_node, innovation_number)

    def mutate_add_node(self):
        if random.random() < ADD_NODE_MUTATE_RATE:
            if not self.connections:
                return

            connection = random.choice(list(self.connections.values()))
            connection.enabled = False

            new_node_gene = HiddenNodeGene(len(self.nodes), NodeType.hidden)
            self.nodes[new_node_gene.key] = new_node_gene

            # TODO: Update global innovation numbers here.
            connection_to_new_node = ConnectionGene(connection.in_node, new_node_gene.key, len(self.connections))
            self.connections[len(self.connections)] = connection_to_new_node

            connection_from_new_node = ConnectionGene(new_node_gene.key, connection.out_node, len(self.connections))
            self.connections[len(self.connections)] = connection_from_new_node
