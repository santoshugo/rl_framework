import networkx as nx
import numpy as np


def create_graph(map):
    nodes, edges, directed = map['nodes'], map['edges'], map['directed']

    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    for node in nodes:
        node_id = node.pop('id')
        graph.add_node(node_id, **node)
        graph.nodes[node_id]['agent'] = None

    for edge in edges:
        node_1 = edge.pop('node_1')
        node_2 = edge.pop('node_2')

        distance = _compute_distance(graph, node_1, node_2)

        graph.add_edge(node_1, node_2, agent=None, distance=distance, **edge)

    return graph


def _compute_distance(graph, node_1, node_2):
    x_diff = graph.nodes[node_1]['x'] - graph.nodes[node_2]['x']
    y_diff = graph.nodes[node_1]['y'] - graph.nodes[node_2]['y']

    return np.sqrt(np.square(x_diff) + np.square(y_diff))

