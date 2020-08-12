"""
implement here functions to help environment/observation logic
"""


def bfs(graph: dict, start):
    """
    Computes shortest path in a graph using breadth first search
    :return:
    """
    queue = [start]
    discovered = {start}
    parents = dict()

    while queue:
        v = queue.pop(0)
        for w in graph[v]:
            if w not in discovered:
                discovered.add(w)
                parents[w] = v
                queue.append(w)

    return parents




