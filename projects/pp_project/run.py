import json

from rl_framework.environment.environment import AbstractEnvironment
from rl_framework.environment.observation import AbstractObservation


class Environment(AbstractEnvironment):
    def __init__(self, environment_map, observation_obj):
        super().__init__(environment_map, observation_obj)

    def step(self, actions):
        observation, reward = None, None
        return observation, reward


class Observation(AbstractObservation):
    def __init__(self):
        pass

    def get(self, agent):
        pass

    def get_all(self):
        pass


if __name__ == '__main__':
    map_file = '/docs/maps/pp_map.json'
    with open(map_file) as f:
        env_map = json.load(f)

    env = Environment(env_map, Observation)
    obs = env.reset()

    obs, r = env.step({0: 'action'})