from gym.spaces import Discrete, MultiDiscrete, Box, Dict
import json
import numpy as np
import networkx as nx
import tensorflow as tf
from ray.rllib.agents.dqn.distributional_q_tf_model import \
    DistributionalQTFModel
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
from ray.rllib.models.tf import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork


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
        # self.observation_space = MultiDiscrete([5, 2])
        self.observation_space = Dict({
            "action_mask": Box(0, 1, shape=(7,)),
            "avail_actions": Box(0, 1, shape=(5, 2)),
            "real_obs": MultiDiscrete([5, 2])
        })

        self.action_mask = {}

        self.num_moves = 0

        self.done0 = False
        self.done1 = False

    def reset(self):
        self.state = {0: 0, 1: 1}
        self.carrying = {0: False, 1: False}
        self.done0 = False
        self.done1 = False

        return {0: self._obs(0), 1: self._obs(1)}

    def get_avail_actions(self, agent):
        action_mask = np.array([0] * self.action_space.n)
        state = self.state[agent]

        for n in self.graph.adj[state]:
            action_mask[n] = 1

        if agent == 0 and state == 2:
            action_mask[5] = 1
        if agent == 0 and state == 0:
            action_mask[6] = 1
        if agent == 1 and state == 4:
            action_mask[5] = 1
        if agent == 1 and state == 1:
            action_mask[6] = 1

        return action_mask

    def step(self, action_dict):
        print(self.state, action_dict, self.get_avail_actions(1))
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
        return {"action_mask": self.get_avail_actions(agent),
                "avail_actions": np.ones(7),
                "real_obs": np.array([self.state[agent], self.carrying[agent]])
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
        self.action_embed_model = FullyConnectedNetwork(MultiDiscrete([5, 2]), action_space, num_outputs, model_config, name + "_action_embed")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)

        return avail_actions + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()