import numpy as np
from nagi.neat import Genome
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy


def visualize_genome(genome: Genome):
    g, nodes, edges = genome_to_graph(genome)
    pos = get_node_coordinates(genome)
    labels = {node: f"{node}{'↩' if (node, node) in edges else ''}" for node in nodes}
    node_color = ['b' if node < genome.input_size else 'r' if node < genome.input_size + genome.output_size else 'g' for
                  node in nodes]

    nx.draw_networkx(g, pos=pos, with_labels=True, labels=labels, nodes=nodes, node_color=node_color, font_color="w",
                     connectionstyle="arc3, rad=0.1")
    plt.show()


def genome_to_graph(genome: Genome):
    edges = [(connection.origin_node, connection.destination_node)
             for connection in genome.get_enabled_connections()]
    nodes = [key for key in genome.nodes.keys()]

    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g, nodes, edges


def get_node_coordinates(genome: Genome):
    def layer_y_linspace(start, end):
        if number_of_nodes == 1:
            return np.mean((start, end))
        else:
            return np.linspace(start, end, number_of_nodes)

    def sort_by_layers():
        keys_with_layers = list(zip(sorted(genome.nodes.keys()), layers))
        return [key for key, _ in sorted(keys_with_layers, key=lambda tup: tup[1])]

    figure_width = 10
    figure_height = 5
    layers = get_layers(genome)
    x = layers / max(layers) * figure_width
    _, number_of_nodes_per_layer = np.unique(layers, return_counts=True)
    y = np.array([])
    for number_of_nodes in number_of_nodes_per_layer:
        margin = figure_height / (number_of_nodes ** 1.5)
        y = np.r_[y, layer_y_linspace(margin, figure_height - margin)]

    y_coords = {key: y for key, y in zip(sort_by_layers(), y)}
    return {key: (x_coord, y_coords[key]) for key, x_coord in zip(sorted(genome.nodes.keys()), x)}


def get_layers(genome: Genome):
    """
    Traverse wMat by row, collecting layer of all nodes that connect to you (X).
    Your layer is max(X)+1
    """
    adjacency_matrix = get_adjacency_matrix(genome)
    adjacency_matrix[:, genome.input_size: genome.input_size + genome.output_size] = 0
    n_node = np.shape(adjacency_matrix)[0]
    layers = np.zeros(n_node)
    while True:  # Loop until sorting doesn't help any more
        prev_order = np.copy(layers)
        for curr in range(n_node):
            src_layer = np.zeros(n_node)
            for src in range(n_node):
                src_layer[src] = layers[src] * adjacency_matrix[src, curr]
            layers[curr] = np.max(src_layer) + 1
        if all(prev_order == layers):
            break
    set_final_layers(layers, genome.input_size, genome.output_size)
    return layers


def get_adjacency_matrix(genome: Genome):
    n = len(genome.nodes)
    node_order_map = {key: i for i, key in enumerate(sorted(genome.nodes.keys()))}
    adjacency_matrix = np.zeros((n, n))
    genome_copy = deepcopy(genome)
    connections_to_ignore = get_last_connection_in_all_cycles(genome_copy)

    for connection in connections_to_ignore:
        genome_copy.connections.pop(connection)
    for connection in genome_copy.get_enabled_connections():
        adjacency_matrix[node_order_map[connection.origin_node]][node_order_map[connection.destination_node]] = 1
    return adjacency_matrix


def get_last_connection_in_all_cycles(genome: Genome):
    return set([max(cycle) for cycle in get_simple_cycles(genome)])


def get_simple_cycles(genome: Genome):
    def cycle_to_list_of_tuples(cycle):
        cycle.append(cycle[0])
        return [(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)]

    edge_to_innovation_number_map = {(connection.origin_node, connection.destination_node): connection.innovation_number
                                     for connection in genome.connections.values()}
    simple_cycles = [cycle_to_list_of_tuples(cycle) for cycle in nx.simple_cycles(genome_to_graph(genome)[0])]
    return [[edge_to_innovation_number_map[edge] for edge in cycle] for cycle in simple_cycles]


def set_final_layers(layers: np.ndarray, input_size: int, output_size: int):
    max_layer = max(layers) + 1
    for i in range(len(layers)):
        if i < input_size:
            layers[i] = 1
        elif i < input_size + output_size:
            layers[i] = max_layer
        elif layers[i] == 1:
            layers[i] = 2

