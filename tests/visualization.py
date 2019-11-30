import networkx as nx
from nagi.neat import Genome
from nagi.visualization import get_node_coordinates
from itertools import count
import matplotlib.pyplot as plt

test_genome = Genome(0, 3, 2, count(0), is_initial_genome=True)
for i in range(10):
    test_genome.mutate()

nodes = [key for key in test_genome.nodes.keys()]
node_color = ['b' if node < 3 else 'r' if node < 5 else 'g' for node in nodes]

edges = [(connection.origin_node, connection.destination_node) for connection in test_genome.connections.values()]

g = nx.DiGraph()
g.add_nodes_from(nodes)
g.add_edges_from(edges)
pos = get_node_coordinates(test_genome)
nx.draw_networkx(g, pos=pos, with_labels=True, nodes=nodes, node_color=node_color, font_color="w")
plt.show()
