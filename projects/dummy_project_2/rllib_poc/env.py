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


PICKUP_REFILL_PROBABILITY = {0: 1, 3: 1, 6: 1, 7: 1, 8: 1}
PENALTY = -100
MOVE_PENALTY = -1
PICKUP_REWARD = 10
DROPDOWN_REWARD = 20
CHARGE_REWARD = 0.1

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

    graph = create_graph(env_map)
    pickup_refill_probability = PICKUP_REFILL_PROBABILITY

    def __init__(self, env_config):
        self.num_agents = env_config['num_agents']
        self.agent_speed = env_config['agent_speed']
        self.initial_node = env_config['initial_node']

        self.agents = {n: ZalandoAgent(n, self, battery_decay_function, battery_charge_function) for n in range(self.num_agents)}

        self.pickup_carts = {0: 5, 3: 5, 6: 5, 7: 5, 8: 5}
        self.charging_station_carts = {1: 0, 5: 0}

    def reset(self):
        for n, agent in self.agents.items():
            agent.set_state(self.initial_node[n], 'node', None)
            agent.set_battery(1)

        self.pickup_carts = {0: 5, 3: 5, 6: 5, 7: 5, 8: 5}
        agent_states = [agent.state for agent in self.agents.values()]

        self.charging_station_carts[1] = agent_states.count(1)
        self.charging_station_carts[5] = agent_states.count(5)

    def step(self, action_dict):
        # refill pickup nodes
        self._refill_pickup()

        obs, rew, done = {}, {}, {}
        for agent_num, action in action_dict.items():
            agent = self.agents[agent_num]
            agent_obs, agent_r, agent_done = agent.step(action)

            obs[agent_num] = agent_obs
            rew[agent_num] = agent_r
            done[agent_num] = agent_done

        done['__all__'] = all(done.values())

        return obs, rew, done, {}

    def _refill_pickup(self):
        """
        Refills pickup spaces
        :return:
        """
        for pickup_station in self.pickup_carts.keys():
            if np.random.random() < self.pickup_refill_probability[pickup_station]:
                self.pickup_carts[pickup_station] = min(5, self.pickup_carts[pickup_station] + 1)


class ZalandoAgent:
    def __init__(self, id, env, decay_function, charge_function):
        self.id = id
        self.graph = graph

        self.state = None
        self.state_type = None
        self.edge_distance = None

        self.env = env
        self.graph = env.graph

        self.battery = 1
        self.decay_function = decay_function
        self.charge_function = charge_function
        self.speed = env.agent_speed

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

    def set_battery(self, battery_charge):
        self.battery = battery_charge

    def get_available_actions(self):
        if self.state_type == 'edge':
            return {10}
        else:
            available_actions = set()
            for n in self.graph.adj[self.state]:
                available_actions.add(n)
            if self.state in {1, 5}:  # charging states
                available_actions.add(9)
            if self.state in {6, 7, 8, 0, 3}:  # pickup states
                available_actions.add(11)
            if self.state in {2, 4}:  # dropdown states
                available_actions.add(12)

    def step(self, action):
        if self.battery <= 0:
            r = PENALTY
        elif action not in self.get_available_actions():  # action ont available
            r = PENALTY
        elif action == 9:  # TODO no more than 5 AGVs at the same charging station
            self.charge()
            r = CHARGE_REWARD
        elif action == 11:  # TODO can't pickup if there is less than 1 item to pick
            self.decay()
            if any([self.carrying_full, self.carrying_empty]):
                r = PENALTY
            elif self.state in [6, 7, 8]:
                self.carrying_full = True
                self.env.pickup_carts[self.state] -= 1
                r = PICKUP_REWARD
            elif self.state in [0, 3]:
                self.carrying_empty = True
                self.env.pickup_carts[self.state] -= 1
                r = PICKUP_REWARD
        elif action == 12:
            self.decay()
            if self.state == 2 and self.carrying_full:
                self.carrying_full = False
                r = DROPDOWN_REWARD
            elif self.state == 4 and self.carrying_empty:
                self.carrying_empty = False
                r = DROPDOWN_REWARD
            else:
                r = PENALTY
        elif action == 10:  # TODO implement movement in edges
            r = MOVE_PENALTY
        else:  # move to other nodes
            self.decay()
            r = MOVE_PENALTY
            self.set_state(action, 'node', None)

        return self.obs, r, False

    def obs(self):
        return np.array([])


graph = create_graph(env_map)
print(graph.nodes)
print(graph.edges)
print(graph.adj[1])