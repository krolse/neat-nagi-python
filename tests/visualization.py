import networkx as nx
from nagi.neat import Genome
from itertools import count
import matplotlib.pyplot as plt

test_genome = Genome(0, 3, 2, count(0), is_initial_genome=True)

nodes = [key for key in test_genome.nodes.keys()]
edges = [(connection.in_node, connection.out_node) for connection in test_genome.connections.values()]

g = nx.DiGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = {0: [0, 2],
       1: [0, 1],
       2: [0, 0],
       3: [1, 0.5],
       4: [1, 1.5]}
nx.draw(g, with_labels=True, pos=pos, font_weight='bold')
plt.show()
