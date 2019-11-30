import numpy as np
import random
from enum import Enum
from copy import deepcopy
from itertools import count, combinations_with_replacement
from typing import List, Dict, Iterator

from nagi.constants import ENABLE_MUTATE_RATE, ADD_CONNECTION_MUTATE_RATE, ADD_NODE_MUTATE_RATE, \
    CONNECTIONS_DISJOINT_COEFFICIENT, CONNECTIONS_EXCESS_COEFFICIENT, INHIBITORY_MUTATE_RATE, LEARNING_RULE_MUTATE_RATE, \
    PREDETERMINED_DISABLED_RATE, INITIAL_CONNECTION_RATE, SPECIES_COMPATIBILITY_THRESHOLD


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

    def mutate(self):
        pass


class HiddenNodeGene(NodeGene):
    # TODO: Should the hidden node gene have a bias, or should the bias be randomly initialized like weights?
    def __init__(self, key: int, is_inhibitory: bool = False,
                 learning_rule: LearningRule = LearningRule.asymmetric_hebbian):
        super().__init__(key, NodeType.hidden)
        self.is_inhibitory = is_inhibitory
        self.learning_rule = learning_rule

    def mutate(self):
        if np.random.random() < INHIBITORY_MUTATE_RATE:
            self.is_inhibitory = not self.is_inhibitory
        if np.random.random() < LEARNING_RULE_MUTATE_RATE:
            self.learning_rule = random.choice([rule for rule in LearningRule if rule is not self.learning_rule])


class OutputNodeGene(NodeGene):
    def __init__(self, key: int, learning_rule: LearningRule = LearningRule.asymmetric_hebbian):
        super().__init__(key, NodeType.output)
        self.is_inhibitory = False
        self.learning_rule = learning_rule

    def mutate(self):
        if np.random.random() < LEARNING_RULE_MUTATE_RATE:
            self.learning_rule = random.choice([rule for rule in LearningRule if rule is not self.learning_rule])


class ConnectionGene(object):
    def __init__(self, origin_node: int, destination_node: int, innovation_number: int):
        self.origin_node = origin_node
        self.destination_node = destination_node
        self.innovation_number = innovation_number
        self.enabled = True

    def mutate(self):
        if np.random.random() < ENABLE_MUTATE_RATE:
            self.enabled = not self.enabled


class Genome(object):
    def __init__(self, key: int, input_size: int, output_size: int, innovation_number_counter: Iterator,
                 is_initial_genome: bool = False):
        self.key = key
        self.innovation_number_counter = innovation_number_counter
        self.nodes = {}
        self.connections = {}
        self.input_size = input_size
        self.output_size = output_size

        input_keys = [i for i in range(input_size)]
        output_keys = [i for i in range(input_size, input_size + output_size)]

        # Initialize node genes for inputs and outputs.
        # TODO: Should probably change this so that all input and output nodes are always inherited.
        for input_key in input_keys:
            self.nodes[input_key] = NodeGene(input_key, NodeType.input)
            for output_key in output_keys:
                if self.nodes.get(output_key) is None:
                    self.nodes[output_key] = OutputNodeGene(output_key)

        # Initialize some connection genes if it is an initial genome.
        if is_initial_genome:  # Guarantee at least one connection to each output node.
            for i, output_key in enumerate(output_keys):
                input_key = random.choice(input_keys)
                innovation_number = input_key * output_size + i
                self.connections[innovation_number] = ConnectionGene(input_key, output_key, innovation_number)

            # Add additional initial connections.
            for input_key in input_keys:
                for i, output_key in enumerate(output_keys):
                    innovation_number = input_key * output_size + i
                    if self.connections.get(innovation_number) is None and random.random() < INITIAL_CONNECTION_RATE:
                        self.connections[innovation_number] = ConnectionGene(input_key, output_key, innovation_number)

    def _mutate_add_connection(self):
        if random.random() < ADD_CONNECTION_MUTATE_RATE:
            (origin_node, destination_node) = random.choice(
                [(origin_node.key, destination_node.key)
                 for (origin_node, destination_node) in combinations_with_replacement(self.nodes.values(), 2)
                 if (origin_node.key, destination_node.key)
                 not in [(connection.origin_node, connection.destination_node)
                         for connection in self.connections.values()]
                 and destination_node.node_type is not NodeType.input])

            innovation_number = next(self.innovation_number_counter)
            self.connections[innovation_number] = ConnectionGene(origin_node, destination_node, innovation_number)

    def _mutate_add_node(self):
        if random.random() < ADD_NODE_MUTATE_RATE:
            if not self.connections:
                return

            connection = random.choice(list(self.connections.values()))
            connection.enabled = False

            new_node_gene = HiddenNodeGene(len(self.nodes))
            self.nodes[new_node_gene.key] = new_node_gene

            innovation_number = next(self.innovation_number_counter)
            connection_to_new_node = ConnectionGene(connection.origin_node, new_node_gene.key, innovation_number)
            self.connections[innovation_number] = connection_to_new_node

            innovation_number = next(self.innovation_number_counter)
            connection_from_new_node = ConnectionGene(new_node_gene.key, connection.destination_node, innovation_number)
            self.connections[innovation_number] = connection_from_new_node

    def mutate(self):
        self._mutate_add_node()
        self._mutate_add_connection()
        for node in self.nodes.values():
            node.mutate()
        for connection in self.connections.values():
            connection.mutate()

    def crossover(self, other, child):
        for key, connection_parent_1 in self.connections.items():
            connection_parent_2 = other.connections.get(key)
            chosen_parent = random.choice([self, other]) if connection_parent_2 is not None else self
            child.connections[key] = deepcopy(chosen_parent.connections[key])
            if self._disabled_connection_gene_in_either_parent(other, key):
                child.connections[key].enabled = False if random.random() < PREDETERMINED_DISABLED_RATE else True

            origin_node_key = chosen_parent.connections[key].origin_node
            destination_node_key = chosen_parent.connections[key].destination_node
            if not child.nodes.get(origin_node_key):
                child.nodes[origin_node_key] = deepcopy(chosen_parent.nodes[origin_node_key])
            if not child.nodes.get(destination_node_key):
                child.nodes[origin_node_key] = deepcopy(chosen_parent.nodes[origin_node_key])

    def _disabled_connection_gene_in_either_parent(self, other, key):
        def check_disabled_connection(connection):
            return connection is not None and not connection.enabled

        connection_1, connection_2 = self.connections.get(key), other.connections.get(key)
        return check_disabled_connection(connection_1) or check_disabled_connection(connection_2)

    def innovation_range(self) -> int:
        return max([key for key in self.connections.keys()])

    def _get_number_of_disjoint_and_excess_connections(self, other):
        disjoint_connections = 0
        excess_connections = 0
        nonmatches = set.union({key for key in self.connections.keys() if key not in other.connections.keys()},
                               {key for key in other.connections.keys() if key not in self.connections.keys()})
        for key in nonmatches:
            if key <= self.innovation_range() and key <= other.innovation_range():
                disjoint_connections += 1
            else:
                excess_connections += 1
        return disjoint_connections, excess_connections

    def distance(self, other):
        d, e = self._get_number_of_disjoint_and_excess_connections(other)
        n = max({len(self.connections), len(other.connections)})
        return (CONNECTIONS_DISJOINT_COEFFICIENT * d + CONNECTIONS_EXCESS_COEFFICIENT * e) / n


class Species(object):
    def __init__(self, key: int, members: List[Genome] = None, representative: Genome = None):
        self.key = key
        self.members = members if members is not None else []
        self.representative = representative

    def add_member(self, specimen: Genome):
        self.members.append(specimen)

    def choose_random_representative(self):
        self.representative = random.choice(self.members)


class Population(object):
    def __init__(self, population_size: int, input_size: int, output_size: int):
        self.genomes: Dict[int, Genome] = {}
        self.species: Dict[int, Species] = {}
        self.genome_id_counter = count(0)
        self.species_id_counter = count(0)
        self.innovation_number_counter = count(input_size * output_size + 1)
        self._input_size = input_size
        self._output_size = output_size

        # Create initial population.
        for _ in range(population_size):
            genome_id = next(self.genome_id_counter)
            self.genomes[genome_id] = Genome(genome_id, self._input_size, self._output_size,
                                             self.innovation_number_counter, is_initial_genome=True)
        # Create initial species.
        self.speciate()

    def speciate(self):
        # Remove any individuals that didn't make it from the previous generation.
        for species in self.species.values():
            species.members = [member for member in species.members if member in self.genomes.values()]

        # Assign species to new individuals.
        unspeciated = [individual for individual in self.genomes.values() if
                       individual not in [member for spec in self.species.values() for member in spec.members]]
        self._assign_species(unspeciated)

        # Choose random representative for the next generation.
        for species in self.species.values():
            species.choose_random_representative()

    def _assign_species(self, unspeciated: List[Genome]):
        for specimen in unspeciated:
            species_assigned = False
            for species in self.species.values():
                if specimen.distance(species.representative) < SPECIES_COMPATIBILITY_THRESHOLD:
                    species.add_member(specimen)
                    species_assigned = True
                    break
            if not species_assigned:
                new_species_id = next(self.species_id_counter)
                self.species[new_species_id] = Species(new_species_id, members=[specimen], representative=specimen)

    def create_new_offspring(self, parent_1: Genome, parent_2: Genome, fitness_1: float, fitness_2: float):
        if fitness_2 > fitness_1:
            parent_1, parent_2 = parent_2, parent_1
        offspring = Genome(next(self.genome_id_counter), self._input_size, self._output_size,
                           self.innovation_number_counter)
        parent_1.crossover(parent_2, offspring)
        return offspring
