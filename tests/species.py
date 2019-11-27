from nagi.neat import Population, Genome
from itertools import count

pop = Population(100, 3, 2)
for species in pop.species.values():
    print(f'ID: {species.key}, number of individuals: {len(species.members)}')
