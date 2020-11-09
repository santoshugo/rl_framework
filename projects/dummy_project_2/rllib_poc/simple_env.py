from gym.spaces import Discrete, MultiDiscrete
import json
import numpy as np
import networkx as nx

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE


class SimpleEnvironment(MultiAgentEnv):
    PICKUP = 5
    DROPDOWN = 6
    MOVEMENT = (0, 1, 2, 3, 4)
    """
    Simple graph environment of the form

    2 -- 3 -- 4
        | |
    0 --| | -- 1

    Agents 1 starts in 1 and goes to 3 (and back), and agent 2 starts in 2 and goes to 5 (and back). Must pickup in 3/5 and dropdown in 1/2
    """
    def __init__(self, config):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([0, 1, 2, 3, 4])
        self.graph.add_edges_from([(0, 3), (1, 3), (2, 3), (4, 3)])

        self.state = {0: 0, 1: 1}
        self.carrying = {0: False, 1: False}

        self.action_space = Discrete(7)
        self.observation_space = MultiDiscrete([5, 2])
        self.num_moves = 0

        self.done0 = False
        self.done1 = False

    def reset(self):
        self.state = {0: 0, 1: 1}
        self.carrying = {0: False, 1: False}
        self.done0 = False
        self.done1 = False

        return {0: self._obs(0), 1: self._obs(1)}

    def step(self, action_dict):
        rew = {}
        obs = {}
        for agent, action in action_dict.items():
            if agent == 0:
                r = self.act_0(action)
            else:
                r = self.act_1(action)

            rew[agent] = r
            obs[agent] = self._obs(agent)

        self.num_moves += 1
        done = {0: self.done0, 1: self.done1, "__all__": all([self.done0, self.done1])}
        # print(action_dict, self.state, self.carrying, rew, done)
        return obs, rew, done, {}

    def act_0(self, action0):
        state0 = self.state[0]

        # movement cases
        if action0 not in self.graph.adj[state0] and action0 not in [self.PICKUP, self.DROPDOWN]:
            r0 = -1
        elif action0 in self.graph.adj[state0]:
            self.state[0] = action0
            r0 = -0.1
        # pickup cases
        elif action0 == 5:
            if state0 != 2:
                r0 = -1
            elif state0 == 2 and not self.carrying[0]:
                self.carrying[0] = True
                r0 = 1
            elif state0 == 2 and self.carrying[0]:
                r0 = -1
        # dropdown cases
        elif action0 == 6:
            if state0 != 0:
                r0 = -1
            elif state0 == 0 and self.carrying[0]:
                self.done0 = True
                self.carrying[0] = False
                r0 = 1
            elif state0 == 0 and not self.carrying[0]:
                r0 = -1

        return r0

    def act_1(self, action1):
        state1 = self.state[1]

        # movement cases
        if action1 not in self.graph.adj[state1] and action1 not in [self.PICKUP, self.DROPDOWN]:
            r1 = -1
        elif action1 in self.graph.adj[state1]:
            self.state[1] = action1
            r1 = -0.1
        # pickup cases
        elif action1 == 5:
            if state1 != 4:
                r1 = -1
            elif state1 == 4 and not self.carrying[1]:
                self.carrying[1] = True
                r1 = 1
            elif state1 == 4 and self.carrying[1]:
                r1 = -1
        # dropdown cases
        elif action1 == 6:
            if state1 != 1:
                r1 = -1
            elif state1 == 1 and self.carrying[1]:
                self.done1 = True
                self.carrying[1] = False
                r1 = 1
            elif state1 == 1 and not self.carrying[1]:
                r1 = -1

        return r1

    def _obs(self, agent):
        return np.array([self.state[agent], self.carrying[agent]])


