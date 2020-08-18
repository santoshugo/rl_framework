import json

from rl_framework.environment.environment import AbstractEnvironment, InfiniteGraphEnvironment
from rl_framework.environment.observation import AbstractObservation


class Environment(InfiniteGraphEnvironment):
    def __init__(self, environment_map, observation_obj):
        super().__init__(environment_map, observation_obj)

    def step(self, actions):
        observation, reward = None, None
        return observation, reward

    def reset(self):
        observation = None
        return observation


class Observation(AbstractObservation):
    def __init__(self):
        pass

    def get(self, agent):
        pass

    def get_all(self):
        pass


if __name__ == '__main__':
    map_file = '/home/hugo/PycharmProjects/rl_framework/docs/maps/pp_map.json'
    with open(map_file) as f:
        env_map = json.load(f)

    env = Environment(env_map, Observation)
    print(env.graph.number_of_nodes())
    print(env.graph.nodes.data())
    print(env.graph.edges.data())


    obs = env.reset()

    obs, r = env.step({0: 'action'})