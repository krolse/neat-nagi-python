import pickle
from itertools import count

from nagi.neat import Genome
from nagi.visualization import visualize_genome

input_size, output_size = 2, 2
test_genome = Genome(0, input_size, output_size, count(input_size + output_size + 1), is_initial_genome=True)
for _ in range(5):
    test_genome.mutate()
visualize_genome(test_genome)

with open('../data/test_genome.pkl', 'wb') as file:
    pickle.dump(test_genome, file)
