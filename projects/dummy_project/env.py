
import numpy as np
import tensorflow as tf

from gym.spaces import Tuple, Discrete, Box, Dict

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel

from rl_framework.environment import AbstractAgent
from rl_framework.utils.graph import from_json


MAP_PATH = 'C:\\Users\\santo\\rl_framework\\docs\maps\\dummy_map_2.json'


def battery_decay_function(x):
    # return x - 100 / (90 * 60)
    return x - 0.05


def battery_charge_function(x):
    # return x + 100 / (180 * 60)
    return x + 0.05


PICKUP_REFILL_PROBABILITY = {0: 1, 3: 1, 6: 1, 7: 1, 8: 1}
PENALTY = -100
MOVE_PENALTY = -1
PICKUP_REWARD = 1000
DROPDOWN_REWARD = 2000
CHARGE_REWARD = 10

initial_state = {0: (1, 'node'),
                 1: (5, 'node')
                 }


class ZalandoEnvironment(MultiAgentEnv):
    action_space = Discrete(9 + 4)  # move to nodes 0-8 and charge (9), move in edge (10), pickup (11) and dropdown (12)

    observation_space = Dict({
        "action_mask": Box(low=0, high=1, shape=(13,)),
        "avail_actions": Box(low=0, high=1, shape=(13,)),
        "real_obs": Box(low=0, high=1, shape=(9 + 1 + 12 + 1 + 2 + 2 + 1,))
    })

    graph = from_json(MAP_PATH)
    pickup_refill_probability = PICKUP_REFILL_PROBABILITY

    def __init__(self, env_config):
        self.num_agents = env_config['num_agents']
        self.agent_speed = env_config['agent_speed']
        self.initial_node = env_config['initial_node']

        self.agents = {n: ZalandoAgent(n, self, battery_decay_function, battery_charge_function) for n in range(self.num_agents)}

        self.pickup_carts = {0: 5, 3: 5, 6: 5, 7: 5, 8: 5}
        self.charging_station_carts = {1: 0, 5: 0}

        self.no_steps = 0

    def reset(self):
        self.no_steps = 0
        for n, agent in self.agents.items():
            agent.set_state({'state': self.initial_node[n], 'state_type': 'node', 'edge_distance': None})
            agent.set_battery(1)

        self.pickup_carts = {0: 5, 3: 5, 6: 5, 7: 5, 8: 5}
        agent_states = [agent.state for agent in self.agents.values()]

        self.charging_station_carts[1] = agent_states.count(1)
        self.charging_station_carts[5] = agent_states.count(5)

        obs = {}
        for n, agent in self.agents.items():
            obs[n] = agent.obs()

        return obs

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
        self.no_steps += 1

        return obs, rew, done, {}

    def _refill_pickup(self):
        """
        Refills pickup spaces
        :return:
        """
        for pickup_station in self.pickup_carts.keys():
            if np.random.random() < self.pickup_refill_probability[pickup_station]:
                self.pickup_carts[pickup_station] = min(5, self.pickup_carts[pickup_station] + 1)


class ZalandoAgent(AbstractAgent):
    def __init__(self, id, environment, decay_function, charge_function):
        super().__init__(id, environment)

        self.state = None
        self.state_type = None
        self.edge_distance = None

        self.graph = self.environment.graph

        self.battery = 1
        self.decay_function = decay_function
        self.charge_function = charge_function
        self.speed = self.environment.agent_speed

        self.carrying_empty = False
        self.carrying_full = False

    def charge(self):
        self.battery = min(1, self.charge_function(self.battery))

    def decay(self):
        self.battery = max(0, self.decay_function(self.battery))

    def set_state(self, state):
        self.state = state['state']
        self.state_type = state['state_type']
        self.edge_distance = state['edge_distance']

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
            if self.state in {6, 7, 8} and not self.carrying_empty and not self.carrying_full:
                available_actions.add(11)
            if self.state == 2 and self.carrying_full:
                available_actions.add(12)
            if self.state == 4 and self.carrying_empty:
                available_actions.add(12)

        return available_actions

    def step(self, action):
        # print(self.state, self.carrying_full, self.carrying_empty, action)
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
                self.environment.pickup_carts[self.state] -= 1
                r = PICKUP_REWARD
            elif self.state in [0, 3]:
                self.carrying_empty = True
                self.environment.pickup_carts[self.state] -= 1
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
            self.set_state({'state': action, 'state_type': 'node', 'edge_distance': None})
        return self.obs(), r, False

    def obs(self):
        node_array = np.zeros(10)
        node_array[self.state] = 1

        edge_array = np.zeros(13)
        edge_array[12] = 1

        cf_array = np.zeros(2)
        cf_array[self.carrying_full] = 1

        ce_array = np.zeros(2)
        ce_array[self.carrying_empty] = 1

        battery_array = np.array([self.battery])

        available_actions_array = np.zeros(13)
        for action in self.get_available_actions():
            available_actions_array[action] = 1

        return {"action_mask": available_actions_array,
                "avail_actions": np.ones(13),
                "real_obs": np.concatenate([node_array, edge_array, cf_array, ce_array, battery_array])
                }


class ParametricActionsModel(DistributionalQTFModel):
    """Parametric action model that handles the dot product and masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kw):
        super(ParametricActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name, **kw)
        self.action_embed_model = FullyConnectedNetwork(Box(low=0, high=1, shape=(9 + 1 + 12 + 1 + 2 + 2 + 1,)), action_space, 13, model_config,
                                                        name + "_action_embed")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embedding, _ = self.action_embed_model({"obs": input_dict["obs"]["real_obs"]})

        intent_vector = tf.expand_dims(action_embedding, 1)
        action_logits = tf.reduce_sum(avail_actions * intent_vector, axis=1)

        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()
