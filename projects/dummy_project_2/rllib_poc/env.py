from gym.spaces import Discrete

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE


class ZalandoEnv(MultiAgentEnv):
    action_space = None

    def __init__(self, env_config):
        self.state = None