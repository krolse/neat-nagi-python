import random
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from itertools import count
from typing import List, Dict, Iterator

import numpy as np

from nagi.constants import ENABLE_MUTATE_RATE, ADD_CONNECTION_MUTATE_RATE, ADD_NODE_MUTATE_RATE, \
    CONNECTIONS_DISJOINT_COEFFICIENT, CONNECTIONS_EXCESS_COEFFICIENT, INHIBITORY_MUTATE_RATE, \
    LEARNING_RULE_MUTATE_RATE, PREDETERMINED_DISABLED_RATE, INITIAL_CONNECTION_RATE, SPECIES_COMPATIBILITY_THRESHOLD, \
    MATING_CUTTOFF_PERCENTAGE, ELITISM, STDP_PARAMETERS_MUTATE_RATE, STDP_PARAMETERS_REINIT_RATE, \
    INHIBITORY_PROBABILITIES, EXCITATORY_PROBABILITIES, SYMMETRIC_A_PLUS_INIT_RANGE, SYMMETRIC_A_MINUS_INIT_RANGE, \
    SYMMETRIC_STD_PLUS_INIT_RANGE, ASYMMETRIC_A_INIT_RANGE, ASYMMETRIC_TAU_INIT_RANGE, SYMMETRIC_A_PLUS_MUTATE_SCALE, \
    SYMMETRIC_A_MINUS_MUTATE_SCALE, SYMMETRIC_STD_PLUS_MUTATE_SCALE, ASYMMETRIC_A_MUTATE_SCALE, \
    ASYMMETRIC_TAU_MUTATE_SCALE, SPECIES_PROTECTION_LIMIT, SPECIES_STAGNATION_LIMIT, SYMMETRIC_STD_MINUS_INIT_RANGE, \
    SYMMETRIC_STD_MINUS_MUTATE_SCALE


class LearningRule(Enum):
    asymmetric_hebbian = 'AH'
    asymmetric_anti_hebbian = 'AA'
    symmetric_hebbian = 'SH'
    symmetric_anti_hebbian = 'SA'

    def is_symmetric(self) -> bool:
        return self is LearningRule.symmetric_hebbian or self is LearningRule.symmetric_anti_hebbian


class NodeGene(ABC):
    def __init__(self, key: int):
        self.key = key

    def mutate(self):
        pass


class NeuralNodeGene(NodeGene):
    @abstractmethod
    def __init__(self, key: int, is_inhibitory: bool):
        super().__init__(key)
        self.is_inhibitory = is_inhibitory
        self.learning_rule = self._initialize_learning_rule()
        self.stdp_parameters = self._initialize_stdp_parameters()

    @abstractmethod
    def mutate(self):
        if np.random.random() < LEARNING_RULE_MUTATE_RATE:
            previous_learning_rule = self.learning_rule
            self.learning_rule = random.choice([rule for rule in LearningRule if rule is not self.learning_rule])
            if previous_learning_rule.is_symmetric() ^ self.learning_rule.is_symmetric():
                self.stdp_parameters = self._initialize_stdp_parameters()
                return

        r = np.random.random()
        if r < STDP_PARAMETERS_MUTATE_RATE:
            self._mutate_stdp_parameters()
        elif r < STDP_PARAMETERS_MUTATE_RATE + STDP_PARAMETERS_REINIT_RATE:
            self.stdp_parameters = self._initialize_stdp_parameters()

    def _initialize_learning_rule(self) -> LearningRule:
        p = INHIBITORY_PROBABILITIES if self.is_inhibitory else EXCITATORY_PROBABILITIES
        return np.random.choice([learning_rule for learning_rule in LearningRule], p=p)

    def _initialize_stdp_parameters(self) -> Dict[str, float]:
        if self.learning_rule.is_symmetric():
            return NeuralNodeGene._initialize_symmetric_stdp_parameters()
        else:
            return NeuralNodeGene._initialize_asymmetric_stdp_parameters()

    def _mutate_stdp_parameters(self):
        if self.learning_rule.is_symmetric():
            self._mutate_symmetric_stdp_parameters()
        else:
            self._mutate_asymmetric_stdp_parameters()

    @staticmethod
    def _initialize_symmetric_stdp_parameters():
        a_plus = np.random.uniform(*SYMMETRIC_A_PLUS_INIT_RANGE)
        a_minus = np.random.uniform(*SYMMETRIC_A_MINUS_INIT_RANGE)
        std_plus = np.random.uniform(*SYMMETRIC_STD_PLUS_INIT_RANGE)
        std_minus = np.random.uniform(*SYMMETRIC_STD_MINUS_INIT_RANGE)
        return {'a_plus': a_plus, 'a_minus': a_minus, 'std_plus': std_plus, 'std_minus': std_minus}

    @staticmethod
    def _initialize_asymmetric_stdp_parameters():
        a_plus, a_minus = np.random.uniform(*ASYMMETRIC_A_INIT_RANGE, 2)
        tau_plus, tau_minus = np.random.uniform(*ASYMMETRIC_TAU_INIT_RANGE, 2)
        return {'a_plus': a_plus, 'a_minus': a_minus, 'tau_plus': tau_plus, 'tau_minus': tau_minus}

    def _mutate_symmetric_stdp_parameters(self):
        a_plus = np.clip(self.stdp_parameters['a_plus'] + np.random.normal(0, SYMMETRIC_A_PLUS_MUTATE_SCALE),
                         *SYMMETRIC_A_PLUS_INIT_RANGE)
        a_minus = np.clip(self.stdp_parameters['a_plus'] + np.random.normal(0, SYMMETRIC_A_MINUS_MUTATE_SCALE),
                          *SYMMETRIC_A_MINUS_INIT_RANGE)
        std_plus = np.clip(self.stdp_parameters['std_plus'] + np.random.normal(0, SYMMETRIC_STD_PLUS_MUTATE_SCALE),
                           *SYMMETRIC_STD_PLUS_INIT_RANGE)
        std_minus = np.clip(self.stdp_parameters['std_plus'] + np.random.normal(0, SYMMETRIC_STD_MINUS_MUTATE_SCALE),
                            *SYMMETRIC_STD_MINUS_INIT_RANGE)
        self.stdp_parameters = {'a_plus': a_plus, 'a_minus': a_minus, 'std_plus': std_plus, 'std_minus': std_minus}

    def _mutate_asymmetric_stdp_parameters(self):
        a_plus, a_minus = (np.clip(self.stdp_parameters[key] + np.random.normal(0, ASYMMETRIC_A_MUTATE_SCALE),
                                   *ASYMMETRIC_A_INIT_RANGE) for key in ('a_plus', 'a_minus'))
        tau_plus, tau_minus = (np.clip(self.stdp_parameters[key] + np.random.normal(0, ASYMMETRIC_TAU_MUTATE_SCALE),
                                       *ASYMMETRIC_TAU_INIT_RANGE) for key in ('tau_plus', 'tau_minus'))
        self.stdp_parameters = {'a_plus': a_plus, 'a_minus': a_minus, 'tau_plus': tau_plus, 'tau_minus': tau_minus}


class InputNodeGene(NodeGene):
    def __init__(self, key: int):
        super().__init__(key)

    def mutate(self):
        pass


class HiddenNodeGene(NeuralNodeGene):
    def __init__(self, key: int):
        super().__init__(key, is_inhibitory=random.choice((True, False)))

    def mutate(self):
        if np.random.random() < INHIBITORY_MUTATE_RATE:
            self.is_inhibitory = not self.is_inhibitory
        super().mutate()


class OutputNodeGene(NeuralNodeGene):
    def __init__(self, key: int):
        super().__init__(key, is_inhibitory=False)

    def mutate(self):
        super().mutate()


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
        self.nodes: Dict[int, NodeGene] = {}
        self.connections = {}
        self.input_size = input_size
        self.output_size = output_size

        input_keys = [i for i in range(input_size)]
        output_keys = [i for i in range(input_size, input_size + output_size)]

        # Initialize node genes for inputs and outputs.
        # TODO: Should probably change this so that all output nodes are always inherited.
        for input_key in input_keys:
            self.nodes[input_key] = InputNodeGene(input_key)
        for output_key in output_keys:
            self.nodes[output_key] = OutputNodeGene(output_key)

        # Initialize some connection genes if it is an initial genome.
        if is_initial_genome:
            # Guarantee at least one connection to each output node.
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
            possible_choices = [(origin_node.key, destination_node.key)
                                for origin_node in self.nodes.values()
                                for destination_node in self.nodes.values()
                                if (origin_node.key, destination_node.key)
                                not in [(connection.origin_node, connection.destination_node)
                                        for connection in self.connections.values()]
                                and not isinstance(destination_node, InputNodeGene)]
            if possible_choices:
                (origin_node, destination_node) = random.choice(possible_choices)
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
        non_matches = set.union({key for key in self.connections.keys() if key not in other.connections.keys()},
                                {key for key in other.connections.keys() if key not in self.connections.keys()})
        for key in non_matches:
            if key <= self.innovation_range() and key <= other.innovation_range():
                disjoint_connections += 1
            else:
                excess_connections += 1
        return disjoint_connections, excess_connections

    def distance(self, other):
        d, e = self._get_number_of_disjoint_and_excess_connections(other)
        n = max({len(self.connections), len(other.connections)})
        return (CONNECTIONS_DISJOINT_COEFFICIENT * d + CONNECTIONS_EXCESS_COEFFICIENT * e) / n

    def get_enabled_connections(self):
        return [connection for connection in self.connections.values() if connection.enabled]


class Species(object):
    def __init__(self, key: int, members: List[Genome] = None, representative: Genome = None):
        self.key = key
        self.members = members if members is not None else []
        self.representative = representative
        self.age = 0
        self._generations_since_improvement = 0
        self._best_fitness = 0

    def __len__(self):
        return len(self.members)

    def add_member(self, specimen: Genome):
        self.members.append(specimen)

    def choose_random_representative(self):
        self.representative = random.choice(self.members)

    def is_protected(self):
        return self.age < SPECIES_PROTECTION_LIMIT

    def is_stagnant(self):
        return self._generations_since_improvement > SPECIES_STAGNATION_LIMIT

    def update_stagnation(self, fitness: float):
        if fitness > self._best_fitness:
            self._best_fitness = fitness
            self._generations_since_improvement = 0
        else:
            self._generations_since_improvement += 1


class Population(object):
    def __init__(self, population_size: int, input_size: int, output_size: int):
        self.genomes: Dict[int, Genome] = {}
        self.species: Dict[int, Species] = {}
        self._genome_id_to_species_id: Dict[int, int] = {}
        self._genome_id_counter = count(0)
        self._species_id_counter = count(0)
        self._innovation_number_counter = count(input_size * output_size + 1)
        self._input_size = input_size
        self._output_size = output_size
        self._population_size = population_size

        # Create initial population.
        for _ in range(population_size):
            genome_id = next(self._genome_id_counter)
            self.genomes[genome_id] = Genome(genome_id, self._input_size, self._output_size,
                                             self._innovation_number_counter, is_initial_genome=True)
        # Create initial species.
        self.speciate()

    def speciate(self):
        # Remove any individuals that didn't make it from the previous generation.
        self._genome_id_to_species_id = {key: species for key, species in self._genome_id_to_species_id.items() if
                                         key in self.genomes.keys()}
        for species in self.species.values():
            species.members = [member for member in species.members if member in self.genomes.values()]

        # Assign species to new individuals.
        unspeciated = [individual for individual in self.genomes.values() if
                       individual not in [member for species in self.species.values() for member in species.members]]
        self._assign_species(unspeciated)

        # Choose random representative for the next generation.
        for species in self.species.values():
            species.choose_random_representative()

    def _remove_extinct_species(self, fitness_by_species: Dict[int, float], fitnesses: Dict[int, float]):
        sorted_species_by_fitness = [key for key, fitness in sorted(fitness_by_species.items(), key=lambda x: x[1])]
        cutoff = int(np.floor(len(sorted_species_by_fitness) * (2 / 3)))
        extinct_species = [species for species in self.species.values()
                           if (species.is_stagnant()
                               and not species.is_protected()
                               and species.key not in sorted_species_by_fitness[cutoff:])]
        for species in extinct_species:
            for member in species.members:
                fitnesses.pop(member.key)
            self.species.pop(species.key)

    def _assign_species(self, unspeciated: List[Genome]):
        for specimen in unspeciated:
            species_assigned = False
            for species in self.species.values():
                if specimen.distance(species.representative) < SPECIES_COMPATIBILITY_THRESHOLD:
                    species.add_member(specimen)
                    species_assigned = True
                    self._genome_id_to_species_id[specimen.key] = species.key
                    break
            if not species_assigned:
                new_species_id = next(self._species_id_counter)
                self.species[new_species_id] = Species(new_species_id, members=[specimen], representative=specimen)
                self._genome_id_to_species_id[specimen.key] = new_species_id

    def _create_new_offspring(self, parent_1: Genome, parent_2: Genome, fitness_1: float, fitness_2: float) -> Genome:
        if fitness_2 > fitness_1:
            parent_1, parent_2 = parent_2, parent_1
        offspring = Genome(next(self._genome_id_counter), self._input_size, self._output_size,
                           self._innovation_number_counter)
        parent_1.crossover(parent_2, offspring)
        offspring.mutate()
        return offspring

    def next_generation(self, fitnesses: Dict[int, float]):
        def sample_two_parents(members: List[Genome]):
            return random.sample(members, 2) if len(members) > 1 else (random.choice(members), random.choice(members))

        # Remove species going extinct.
        fitness_by_species = self._get_sum_of_adjusted_fitnesses_by_species(fitnesses)
        for key, species in self.species.items():
            species.age += 1
            species.update_stagnation(fitness_by_species[key])
        self._remove_extinct_species(fitness_by_species, fitnesses)

        # Create new population of genomes
        assigned_number_of_offspring_per_species = self._assign_number_of_offspring_to_species(fitnesses)
        new_population_of_genomes = {}
        for species_id, species in self.species.items():
            species_size = assigned_number_of_offspring_per_species[species_id]
            old_members = sorted(species.members, key=lambda x: fitnesses[x.key], reverse=True)

            for genome in old_members[:ELITISM]:
                new_population_of_genomes[genome.key] = genome
                species_size -= 1

            cutoff = max(int(np.ceil(MATING_CUTTOFF_PERCENTAGE * len(old_members))), 2)
            old_members = old_members[:cutoff]

            while species_size > 0:
                species_size -= 1
                parent_1, parent_2 = sample_two_parents(old_members)
                offspring = self._create_new_offspring(parent_1, parent_2,
                                                       fitnesses[parent_1.key],
                                                       fitnesses[parent_2.key])
                new_population_of_genomes[offspring.key] = offspring

        self.genomes = new_population_of_genomes
        self.speciate()

    def _assign_number_of_offspring_to_species(self, fitnesses: Dict[int, float]) -> Dict[int, int]:
        total_adjusted_fitness = self._get_total_sum_of_adjusted_fitnesses(fitnesses)
        sum_of_adjusted_fitnesses_by_species = self._get_sum_of_adjusted_fitnesses_by_species(fitnesses)
        assigned_number_of_offspring = {
            species_id: max(round(species_fitness * self._population_size / total_adjusted_fitness), 2)
            for species_id, species_fitness in sum_of_adjusted_fitnesses_by_species.items()}

        self._tune_assigned_offspring_to_population_size(assigned_number_of_offspring)
        return assigned_number_of_offspring

    def _tune_assigned_offspring_to_population_size(self, assigned_offspring):
        difference = sum(assigned_offspring.values()) - self._population_size
        while difference != 0:
            if difference > 0:
                species_id = max(assigned_offspring.items(), key=lambda x: x[1])[0]
                assigned_offspring[species_id] -= 1
                difference -= 1
            else:
                species_id = min(assigned_offspring.items(), key=lambda x: x[1])[0]
                assigned_offspring[species_id] += 1
                difference += 1

    def _get_species(self, genome_id: int) -> Species:
        return self.species[self._genome_id_to_species_id[genome_id]]

    def _get_fitness_sharing_adjusted_fitnesses(self, fitnesses: Dict[int, float]) -> Dict[int, float]:
        return {genome_id: fitness / len(self._get_species(genome_id)) for genome_id, fitness in fitnesses.items()}

    def _get_sum_of_adjusted_fitnesses_by_species(self, fitnesses: Dict[int, float]) -> Dict[int, float]:
        sum_of_adjusted_fitnesses_by_species = {species_id: 0 for species_id in self.species.keys()}
        for genome_id, adjusted_fitness in self._get_fitness_sharing_adjusted_fitnesses(fitnesses).items():
            sum_of_adjusted_fitnesses_by_species[self._get_species(genome_id).key] += adjusted_fitness
        return sum_of_adjusted_fitnesses_by_species

    def _get_total_sum_of_adjusted_fitnesses(self, fitnesses: Dict[int, float]) -> float:
        return sum(self._get_fitness_sharing_adjusted_fitnesses(fitnesses).values())
