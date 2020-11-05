from gym.spaces import Discrete
import json
import numpy as np
import networkx as nx

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE


class SimpleEnvironment(MultiAgentEnv):
    PICKUP = 6
    DROPDOWN = 7
    MOVEMENT = (1, 2, 3, 4, 5)
    """
    Simple graph environment of the form

    3 -- 4 -- 5
        | |
    1 --| | -- 2

    Agents 1 starts in 1 and goes to 3 (and back), and agent 2 starts in 2 and goes to 5 (and back). Must pickup in 3/5 and dropdown in 1/2
    """
    def __init__(self, config):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([1, 2, 3, 4, 5])
        self.graph.add_edges_from([(1, 4), (2, 4), (3, 4), (4, 5)])

        self.state = {1: 1, 2: 2}
        self.carrying = {1: False, 2: False}

        self.action_space = Discrete(5 + 2)
        self.observation_space = Discrete(5 * 2)  # ?
        self.num_moves = 0

    def reset(self):
        self.num_moves = 0
        self.state = {1: 1, 2: 2}

    def step(self, action_dict):
        state1, action1 = self.state[1], action_dict[1]
        state2, action2 = self.state[2], action_dict[2]

        # movement cases
        if action1 not in self.graph.adj[state1] and action1 not in [self.PICKUP, self.DROPDOWN]:
            r1 = -1
        elif action1 in self.graph.adj[state1]:
            self.state[1] = action1
            r1 = 0
        # pickup cases
        elif action1 == 6:
            if state1 != 3:
                r1 = -1
            elif state1 == 3 and not self.carrying[1]:
                self.carrying[1] = True
                r1 = 1
            elif state1 == 3 and self.carrying[1]:
                r1 = -1
        # dropdown cases
        elif action1 == 7:
            if state1 != 1:
                r1 = -1
            elif state1 == 1 and self.carrying[1]:
                self.carrying[1] = False
                r1 = 1
            elif state1 == 1 and not self.carrying[1]:
                r1 = -1

        # movement cases
        if action2 not in self.graph.adj[state2] and action2 not in [self.PICKUP, self.DROPDOWN]:
            r2 = -1
        elif action2 in self.graph.adj[state2]:
            self.state[2] = action2
            r2 = 0
        # pickup cases
        elif action2 == 6:
            if state2 != 3:
                r2 = -1
            elif state2 == 5 and not self.carrying[2]:
                self.carrying[2] = True
                r2 = 1
            elif state2 == 5 and self.carrying[2]:
                r2 = -1
        # dropdown cases
        elif action2 == 7:
            if state2 != 2:
                r2 = -1
            elif state2 == 2 and self.carrying[2]:
                self.carrying[2] = False
                r2 = 1
            elif state2 == 2 and not self.carrying[2]:
                r2 = -1

        self.num_moves += 1
        done = {"__all__": self.num_moves >= 50}

        return self.state, {1: r1, 2: r2}, done, {}


