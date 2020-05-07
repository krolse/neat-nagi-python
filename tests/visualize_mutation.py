from itertools import count

import matplotlib.pyplot as plt
import networkx as nx

from nagi.neat import Genome
from nagi.visualization import get_node_coordinates

input_size = 6
output_size = 2
test_genome = Genome(0, input_size, output_size, count(input_size * output_size + 1), is_initial_genome=True)

number_of_mutations = 20
n = max([number_of_mutations, 20])

for i in range(1, n + 1):
    if i > n - 20:
        plt.subplot(4, 5, i - (n - 20), title=f'Gen {i}', frame_on=False)
        edges = [(connection.origin_node, connection.destination_node)
                 for connection in test_genome.connections.values()
                 if connection.enabled]
        nodes = [key for key in test_genome.nodes.keys()]
        node_color = ['b' if node < input_size else 'r' if node < input_size+output_size else 'g' for node in nodes]

        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = get_node_coordinates(test_genome)
        labels = {node: f"{node}{'↩' if (node, node) in edges else ''}" for node in nodes}
        nx.draw_networkx(g, pos=pos, with_labels=True, labels=labels, nodes=nodes, node_color=node_color, font_color="w",
                         connectionstyle="arc3, rad=0.1")
    for connection in test_genome.connections.values():
        print((connection.origin_node, connection.destination_node))

    test_genome.mutate()


mng = plt.get_current_fig_manager()
# mng.window.state('zoomed')
plt.show()
