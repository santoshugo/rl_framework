import json

from rl_framework.environment.environment import InfiniteGraphEnvironment
from rl_framework.environment.observation import AbstractObservation


class Environment(InfiniteGraphEnvironment):
    def __init__(self, environment_map, observation_obj, initial_state):
        super().__init__(environment_map, observation_obj, initial_state, malfunction_len=0, malfunction_prob=0)

    def step(self, actions):
        observation, reward = None, None
        return observation, reward


class Observation(AbstractObservation):
    def __init__(self, environment):
        self.environment = environment

    def get(self, agent):
        pass

    def get_all(self):
        pass


if __name__ == '__main__':
    map_file = '/home/hugo/PycharmProjects/rl_framework/docs/maps/pp_map.json'
    with open(map_file) as f:
        env_map = json.load(f)

    initial_node = {0: 1}

    env = Environment(env_map, Observation, initial_node)

    print(env.graph.nodes.data())

    obs = env.reset()
    print(env.graph.nodes.data())

    obs, r = env.step({0: 'action'})