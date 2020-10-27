from gym.spaces import Discrete

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from projects.dummy_project_2.agent import ZalandoAgent


MAP_PATH = 'C:\\Users\\santosh\\PycharmProjects\\rl_framework\\docs\\maps\\dummy_map_2.json'


def battery_decay_function(x):
    return x - 100 / (90 * 60)


def battery_charge_function(x):
    return x + 100 / (180 * 60)


class ZalandoEnv(MultiAgentEnv):
    action_space = None

    def __init__(self, env_config):
        self.state = None
        self.agents = {agent_no: ZalandoAgent(agent_no, 0.5, battery_decay_function, battery_charge_function) for agent_no in range(10)}