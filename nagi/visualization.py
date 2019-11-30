import numpy as np
from nagi.neat import Genome


def get_node_coordinates(genome: Genome):
    def layer_y_linspace(start, end, number_of_nodes):
        if number_of_nodes == 1:
            return np.mean((start, end))
        else:
            return np.linspace(start, end, number_of_nodes)

    def sort_by_layers(l):
        keys_with_layers = list(zip(sorted(genome.nodes.keys()), l))
        return [key for key, _ in sorted(keys_with_layers, key=lambda tup: tup[1])]

    figure_width = 10
    figure_height = 5
    layers = get_layers(genome)
    x = layers/max(layers) * figure_width
    _, number_of_nodes_per_layer = np.unique(layers, return_counts=True)
    y = np.array([])
    for number_of_nodes in number_of_nodes_per_layer:
        y = np.r_[y, layer_y_linspace(0, figure_height, number_of_nodes)]

    y_coords = {key: y for key, y in zip(sort_by_layers(layers), y)}
    return {key: (x_coord, y_coords[key]) for key, x_coord in zip(sorted(genome.nodes.keys()), x)}


def get_layers(genome: Genome):
    """
    Traverse wMat by row, collecting layer of all nodes that connect to you (X).
    Your layer is max(X)+1
    """
    adjacency_matrix = get_adjacency_matrix(genome)
    np.fill_diagonal(adjacency_matrix, 0)
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
    return set_input_output_layer(layers, genome.input_size, genome.output_size)


def get_adjacency_matrix(genome: Genome):
    n = len(genome.nodes)
    node_order_map = {key: i for (i, key) in enumerate(sorted(genome.nodes.keys()))}
    adjacency_matrix = np.zeros((n, n))
    for connection in genome.connections.values():
        adjacency_matrix[node_order_map[connection.origin_node]][node_order_map[connection.destination_node]] = 1
    return adjacency_matrix


def set_input_output_layer(layers: np.ndarray, input_size: int, output_size: int):
    max_layer = max(layers) + 1
    for i in range(input_size):
        layers[i] = 1
    for i in range(input_size, input_size + output_size):
        layers[i] = max_layer
    return layers
