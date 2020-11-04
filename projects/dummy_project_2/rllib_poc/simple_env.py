from gym.spaces import Discrete
import json
import numpy as np
import networkx as nx

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE


class SimpleEnvironment(MultiAgentEnv):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action_dict):
        pass