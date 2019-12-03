from nagi.neat import Population
from random import random

pop = Population(100, 3, 2)

for i in range(10):
    fitnesses = {key: random() for key in pop.genomes.keys()}
    print(f'--Generation {i}:')
    for species in pop.species.values():
        print(f'ID: {species.key}, number of individuals: {len(species)}')

    pop.next_generation({key: random() for key in pop.genomes.keys()})
