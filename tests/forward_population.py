import random

from nagi.neat import Population
from nagi.visualization import visualize_genome

pop = Population(100, 3, 2)

for i in range(30):
    print(f'--Generation {i}:')
    for species in pop.species.values():
        print(f'ID: {species.key}, number of individuals: {len(species)}')

    pop.next_generation({key: random.random() for key in pop.genomes.keys()})
visualize_genome(random.choice(list(pop.genomes.values())))
