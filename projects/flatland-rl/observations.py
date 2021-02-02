from flatland.core.env_observation_builder import ObservationBuilder
from flatland.utils.rendertools import RenderTool
from flatland.envs.rail_env import RailEnv

import networkx as nx

from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List


class SimpleGraphObservation(ObservationBuilder):
    def __init__(self):
        super().__init__()
        self.G = nx.DiGraph()

        self.agent_destinations = None
        self.agent_state = None
        self.no_discrete_states = None

        self.all_states = set()

        self.node_dictionary = {}
        self.edge_dictionary = {}

    def reset(self):
        self.agent_destinations = set([agent.target for agent in self.env.agents])
        self.get_all_states()

        for state in self.all_states:
            if state not in self.node_dictionary.keys() and state not in self.edge_dictionary.keys():
                self.compute_graph(state)

        self.no_discrete_states = max(max(self.node_dictionary.values()), max(self.edge_dictionary.values()))

        self.G = nx.relabel.convert_node_labels_to_integers(self.G)
        for n in self.G.nodes:
            self.node_dictionary[(self.G.nodes[n]['pos'], self.G.nodes[n]['direction'])] = n

        #nx.draw(self.G, pos={n: (self.G.nodes[n]['pos'][1], -self.G.nodes[n]['pos'][0]) for n in self.G.nodes})
        #plt.show()

    def get_many(self, handles: Optional[List[int]] = None):
        observations = {}
        partial_observations = {}

        if handles is None:
            return self.G, observations

        for h in handles:
            partial_observations[h] = self.get(h)

        for h in handles:
            temp = deepcopy(partial_observations)
            self_obs = temp.pop(h)
            if len(handles) == 1:
                observations[h] = {'self': self_obs}
            else:
                observations[h] = {'self': self_obs, 'other': tuple(temp.values())}

        return self.no_discrete_states, self.G, observations

    def get(self, handle: int = 0):
        observation = {}

        agent = self.env.agents[handle]
        if agent.status == 0:
            pos = agent.initial_position
        elif agent.status == 3:
            pos = agent.old_position
        else:
            pos = agent.position

        state = (pos, agent.direction)

        observation['position'] = self.node_dictionary[state] if state in self.node_dictionary.keys() else self.edge_dictionary[state]
        observation['position_type'] = 0 if pos in self.node_dictionary.keys() else 1  # 0 if in node, 1 if in edge
        observation['direction'] = agent.direction
        observation['moving'] = agent.moving
        observation['target'] = 0 # self.node_dictionary[agent.target]

        # TODO add more relevant agent information:
        # - next node when in edge?
        # - how far along the edge
        # - malfunction info
        # - speed info

        return observation

    def compute_graph(self, initial_node):
        # dfs to get graph structure
        node_ind = 1

        discovered = set()
        stack = [initial_node]

        while stack:
            v = stack.pop()
            if v not in discovered:
                discovered.add(v)

                if v not in self.node_dictionary.keys():
                    self.node_dictionary[v] = node_ind
                    node_ind += 1

                self.G.add_node(self.node_dictionary[v], pos=v[0], direction=v[1])

                for w in self.get_neighbors(v):
                    if w not in self.node_dictionary.keys():
                        self.node_dictionary[w] = node_ind
                        node_ind += 1

                    self.G.add_edge(self.node_dictionary[v], self.node_dictionary[w], distance=1, path=[])
                    stack.append(w)

        # merges route of nodes into an edge
        self.simplify_graph()

    def simplify_graph(self):
        edge_id = 0

        removable_nodes = set([n for n in list(self.G.nodes) if self.G.out_degree(n) == 1
                               and self.G.in_degree(n) == 1
                               and not self.env.rail.is_dead_end(self.G.nodes[n]['pos'])
                               and self.G.nodes[n]['pos'] not in self.agent_destinations])

        while removable_nodes:
            n = removable_nodes.pop()

            start_point = None
            end_point = None

            path = list()
            stack = [(n, 'origin')]

            while stack:
                v, path_position = stack.pop()
                if v not in path:
                    path.insert(0, v) if path_position == 'predecessor' else path.append(v)

                    for w in self.G.successors(v):
                        if w in removable_nodes:
                            removable_nodes.remove(w)
                            stack.append((w, 'successor'))
                        else:
                            end_point = w

                    for y in self.G.predecessors(v):
                        if y in removable_nodes:
                            removable_nodes.remove(y)
                            stack.append((y, 'predecessor'))
                        else:
                            start_point = y

            self.G.add_edge(start_point, end_point, path=path, distance=len(path), edge_id=edge_id)

            for n in path:
                self.edge_dictionary[(self.G.nodes[n]['pos'], self.G.nodes[n]['direction'])] = edge_id
                del self.node_dictionary[(self.G.nodes[n]['pos'], self.G.nodes[n]['direction'])]
                self.G.remove_node(n)

            edge_id += 1

    def get_neighbors(self, state):
        h, w, direction = state[0][0], state[0][1], state[1]
        neighbors = []
        transitions = self.env.rail.get_transitions(h, w, direction)

        # move north
        if h > 0 and transitions[0] == 1:
            neighbors.append(((h - 1, w), 0))
        # move south
        if h < self.env.rail.height - 1 and transitions[2] == 1:
            neighbors.append(((h + 1, w), 2))
        # move east
        if w < self.env.rail.width - 1 and transitions[1] == 1:
            neighbors.append(((h, w + 1), 1))
        # move west
        if w > 0 and transitions[3] == 1:  # move west
            neighbors.append(((h, w - 1), 3))

        return neighbors

    def get_all_states(self):
        for h in range(self.env.height):
            for w in range(self.env.width):
                transitions = np.binary_repr(self.env.rail.grid[h, w], 16)
                if transitions[:4].count('1') >= 1:
                    self.all_states.add(((h, w), 0))
                if transitions[4:8].count('1') >= 1:
                    self.all_states.add(((h, w), 1))
                if transitions[8:12].count('1') >= 1:
                    self.all_states.add(((h, w), 2))
                if transitions[12:].count('1') >= 1:
                    self.all_states.add(((h, w), 3))


if __name__ == '__main__':
    env = RailEnv(width=7, height=7, obs_builder_object=SimpleGraphObservation())
    obs = env.reset()
