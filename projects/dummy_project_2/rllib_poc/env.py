from gym.spaces import Tuple, Discrete, Box
import json
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from projects.dummy_project_2.environment import ZalandoObservation
from projects.dummy_project_2.utils import create_graph

MAP_PATH = 'C:\\Users\\santosh\\PycharmProjects\\rl_framework\\docs\\maps\\dummy_map_2.json'
with open(MAP_PATH) as f:
    env_map = json.load(f)


def battery_decay_function(x):
    return x - 100 / (90 * 60)


def battery_charge_function(x):
    return x + 100 / (180 * 60)


initial_state = {0: (1, 'node'),
                 1: (1, 'node'),
                 # 2: (1, 'node'),
                 # 3: (1, 'node'),
                 # 4: (1, 'node'),
                 # 5: (5, 'node'),
                 # 6: (5, 'node'),
                 # 7: (5, 'node'),
                 # 8: (5, 'node'),
                 # 9: (5, 'node')
                 }


class ZalandoEnvironment(MultiAgentEnv):
    action_space = Discrete(9 + 4)  # move to nodes 0-8 and charge (9), move in edge (10), pickup (11) and dropdown (12)
    observation_space = Tuple((Discrete(9 + 1),  # nodes + not in node
                               Discrete(12 + 1),  # edges + not in edge
                               Discrete(2),  # carrying full
                               Discrete(2),  # carrying empty
                               Box(0, 1, shape=(1,))  # battery
                               ))

    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action_dict):
        pass

    def _obs(self):
        pass


class ZalandoAgent:
    def __init__(self, id, speed, decay_function, charge_function):
        self.id = id

        self.state = None
        self.state_type = None
        self.edge_distance = None

        self.graph = None

        self.battery = 1
        self.decay_function = decay_function
        self.charge_function = charge_function
        self.speed = speed

        self.carrying_empty = False
        self.carrying_full = False

    def charge(self):
        self.battery = min(1, self.charge_function(self.battery))

    def decay(self):
        self.battery = max(0, self.decay_function(self.battery))

    def set_state(self, state, state_type, edge_distance):
        self.state = state
        self.state_type = state_type
        self.edge_distance = edge_distance

    def step(self):
        pass

    def obs(self):
        pass


graph = create_graph(env_map)
print(graph.nodes)
print(graph.edges)
print(graph.adj[1])