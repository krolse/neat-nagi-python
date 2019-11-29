import numpy as np

from nagi.neat import Genome


def get_adjacency_matrix(genome: Genome):
    n = len(genome.nodes)
    adjacency_matrix = np.zeros((n, n))
    for connection in genome.connections.values():
        adjacency_matrix[connection.origin_node][connection.destination_node] = 1
    return adjacency_matrix


def get_layer(genome: Genome):
    """
    Traverse wMat by row, collecting layer of all nodes that connect to you (X).
    Your layer is max(X)+1
    """
    adjacency_matrix = get_adjacency_matrix(genome)
    for i in range(len(genome.nodes)):
        adjacency_matrix[i][i] = 0
    adjacency_matrix[np.isnan(adjacency_matrix)] = 0
    adjacency_matrix[adjacency_matrix != 0] = 1
    n_node = np.shape(adjacency_matrix)[0]
    layer = np.zeros(n_node)
    while True:  # Loop until sorting doesn't help any more
        prev_order = np.copy(layer)
        for curr in range(n_node):
            src_layer = np.zeros(n_node)
            for src in range(n_node):
                src_layer[src] = layer[src] * adjacency_matrix[src, curr]
            layer[curr] = np.max(src_layer) + 1
        if all(prev_order == layer):
            break
    return set_input_output_layer(layer - 1, genome.input_size, genome.output_size)


def set_input_output_layer(layers: np.ndarray, input_size: int, output_size: int):
    for i in range(input_size):
        layers[i] = 0
    for i in range(input_size, input_size + output_size):
        layers[i] = max(layers)
    return layers
