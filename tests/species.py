from nagi.neat import Population
from random import random

pop = Population(100, 3, 2)
fitnesses = {key: random() for key in pop.genomes.keys()}
assigned_individuals = pop.assign_number_of_offspring_to_species(fitnesses)
print(sum(assigned_individuals.values()))
for species_id, assigned in assigned_individuals.items():
    print(f'ID: {species_id}, assigned individuals: {assigned}')
for species in pop.species.values():
    print(f'ID: {species.key}, number of individuals: {len(species)}')
