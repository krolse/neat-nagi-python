import numpy as np
import networkx as nx

matrix = np.fromfile('matrix.npy')
g = nx.convert_matrix.from_numpy_array(matrix, create_using=nx.DiGraph)
print(nx.simple_cycles(g))
