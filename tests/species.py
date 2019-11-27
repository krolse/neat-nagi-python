from nagi.neat import Population, Genome
from itertools import count

pop = Population(100, 3, 2)
for species in pop.species.items():
    print(f'ID: {species[0]}, number of individuals: {len(species[1].members)}')
