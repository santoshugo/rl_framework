import networkx as nx
import json


def from_json(path_to_file):
    """
    Imports a json file with map information and creates a networkx graph object. File should have list of nodes and edges, as well as a
    "directed" field. If creating a directed graph direction of edges is from node_1 to node_2.
    Example of json file:

    {
      "nodes": [
        {
          "id": 0,
          "att1": "John",
          "att2": 31
        },
        {
          "id": 1,
          "att1": Jane,
          "att2": 28,
        }
        ],
        "edges": [
        {
          "node_1": 0,
          "node_2": 1,
          "att1": "red"
        }
        ],
      "directed": true
    }

    :param path_to_file: path to json file with map info
    :return: networkx graph object
    """
    with open(path_to_file) as f:
        data = json.load(f)

    nodes, edges, directed = data['nodes'], data['edges'], data['directed']

    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    for node in nodes:
        node_id = node.pop('id')
        graph.add_node(node_id, **node)

    edge_id = 0
    for edge in edges:
        node_1 = edge.pop('node_1')
        node_2 = edge.pop('node_2')

        graph.add_edge(node_1, node_2, id=edge_id, **edge)
        edge_id += 1

    return graph

