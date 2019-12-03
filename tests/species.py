from nagi.neat import Population
from random import random

pop = Population(100, 3, 2)

for i in range(10):
    fitnesses = {key: random() for key in pop.genomes.keys()}
    print(f'--Generation {i}:')
    assigned = pop.assign_number_of_offspring_to_species(fitnesses)
    for species in pop.species.values():
        print(f'ID: {species.key}, number of individuals: {len(species)}')
    for key, value in assigned.items():
        print(f'ID: {key}, assigned offspring: {value}')

    pop.next_generation({key: random() for key in pop.genomes.keys()})
