from itertools import count

from nagi.constants import REGULAR_SPIKING_PARAMS
from nagi.neat import Genome
from nagi.snn import SpikingNeuralNetwork

genome = Genome(0, 3, 2, count(0), True)
network = SpikingNeuralNetwork.create(genome, 0.5, **REGULAR_SPIKING_PARAMS)
pass
