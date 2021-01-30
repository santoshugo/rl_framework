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
        self.G = nx.Graph()

        self.agent_destinations = None
        self.agent_state = None

        self.node_dictionary = {}
        self.edge_dictionary = {}

    def reset(self):
        self.agent_destinations = set([agent.target for agent in self.env.agents])
        self.compute_graph()
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
            observations[h] = {'self': self_obs, 'other': list(temp.values())}

        return self.G, observations

    def get(self, handle: int = 0):
        observation = {}

        agent = self.env.agents[handle]
        pos = agent.position if agent.status != 0 else agent.initial_position

        observation['position'] = self.node_dictionary[pos] if pos in self.node_dictionary.keys() else self.edge_dictionary[pos]
        observation['position_type'] = 0 if pos in self.node_dictionary.keys() else 1  # 0 if in node, 1 if in edge
        observation['direction'] = agent.direction
        observation['moving'] = agent.moving
        observation['target'] = self.node_dictionary[agent.target]

        # TODO add more relevant agent information:
        # - next node when in edge?
        # - how far along the edge
        # - malfunction info
        # - speed info

        return observation

    def compute_graph(self):
        # dfs to get graph structure
        initial_node = self.env.agents[0].initial_position
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

                self.G.add_node(self.node_dictionary[v], pos=v)

                for w in self.get_neighbors(v):

                    if w not in self.node_dictionary.keys():
                        self.node_dictionary[w] = node_ind
                        node_ind += 1

                    self.G.add_edge(self.node_dictionary[v], self.node_dictionary[w], distance=1, path=[])
                    stack.append(w)

        # merges route of nodes into an edge
        self.simplify_graph()

    def simplify_graph(self):
        edge_id = 1
        removable_nodes = set([n for n in list(self.G.nodes) if self.G.degree[n] == 2])
        for n in self.agent_destinations:
            removable_nodes.discard(self.node_dictionary[n])

        while removable_nodes:
            n = removable_nodes.pop()
            end_points = []
            path = set()
            stack = [n]

            while stack:
                v = stack.pop()
                if v not in path:
                    path.add(v)

                    for w in self.G.adj[v]:
                        if w in removable_nodes:
                            removable_nodes.remove(w)
                            stack.append(w)
                        elif self.G.degree[w] == 2 and self.G.nodes[w]['pos'] not in self.agent_destinations:
                            stack.append(w)
                        else:
                            end_points.append(w)

            self.G.add_edge(end_points[0], end_points[1], path=path, distance=len(path), edge_id=edge_id)
            edge_id += 1
            for n in path:
                self.edge_dictionary[self.G.nodes[n]['pos']] = edge_id
                del self.node_dictionary[self.G.nodes[n]['pos']]
                self.G.remove_node(n)

    def get_neighbors(self, position):
        h, w = position
        neighbors = []
        transitions = np.binary_repr(self.env.rail.grid[h, w], 16)

        # move north
        if h > 0:
            north_neighbor = (h - 1, w)
            north_transitions = np.binary_repr(self.env.rail.grid[h - 1, w], 16)

            if transitions[::4].count('1') >= 1:
                neighbors.append(north_neighbor)
            elif north_transitions[2::4].count('1') >= 1:
                neighbors.append(north_neighbor)

        # move south
        if h < self.env.rail.height - 1:
            south_neighbor = (h + 1, w)
            south_transitions = np.binary_repr(self.env.rail.grid[h + 1, w], 16)

            if transitions[2::4].count('1') >= 1:
                neighbors.append(south_neighbor)
            elif south_transitions[::4].count('1') >= 1:
                neighbors.append(south_neighbor)

        # move east
        if w < self.env.rail.width - 1:
            east_neighbor = (h, w + 1)
            east_transitions = np.binary_repr(self.env.rail.grid[h, w + 1], 16)

            if transitions[1::4].count('1') >= 1:
                neighbors.append(east_neighbor)
            elif east_transitions[3::4].count('1') >= 1:
                neighbors.append(east_neighbor)

        # move west
        if w > 0:  # move west
            west_neighbor = (h, w - 1)
            west_transitions = np.binary_repr(self.env.rail.grid[h, w - 1], 16)

            if transitions[3::4].count('1') >= 1:
                neighbors.append(west_neighbor)
            elif west_transitions[1::4].count('1') >= 1:
                neighbors.append(west_neighbor)

        return neighbors


if __name__ == '__main__':
    env = RailEnv(width=8, height=8,
                  number_of_agents=3,
                  obs_builder_object=SimpleGraphObservation())

    obs = env.reset()
    print(obs)
    #obs, r, done, _ = env.step({0: 2})
    #print(obs)
    #obs, r, done, _ = env.step({0: 2})
    #print(obs)

    #env_renderer = RenderTool(env)
    #env_renderer.render_env(show=True, frames=True, show_observations=False)
    #input("Press Enter to continue...")

    # while True:
    #     action = int(input('Action:'))
    #
    #     env.step({0: action})
    #     env_renderer.render_env(show=True, frames=True, show_observations=False)
