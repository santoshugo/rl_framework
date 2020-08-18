import json
from copy import deepcopy
import numpy as np

from rl_framework.environment.environment import InfiniteGridEnvironment
from rl_framework.environment.observation import GlobalObservation


class ZalandoEnvironment(InfiniteGridEnvironment):
    def __init__(self, environment_map, observation_obj, start='full'):
        super().__init__(environment_map, observation_obj, malfunction_prob=0, malfunction_len=0)

    def __build(self):
        self.pickup_full_1 = None
        self.pickup_full_2 = None
        self.pickup_full_3 = None

        self.pickup_empty_1 = None
        self.pickup_empty_2 = None

        self.drop_full_zone = None
        self.drop_empty_zone = None

        self.charging_station_zones = []

        for s in self.environment:
            if s['pickup_full_1']:
                self.pickup_full_1 = (s['x'], s['y'])
            if s['pickup_full_2']:
                self.pickup_full_2 = (s['x'], s['y'])
            if s['pickup_full_3']:
                self.pickup_full_3 = (s['x'], s['y'])
            if s['pickup_empty_1']:
                self.pickup_empty_1 = (s['x'], s['y'])
            if s['pickup_empty_2']:
                self.pickup_empty_2 = (s['x'], s['y'])
            if s['drop_full']:
                self.drop_full_zone = (s['x'], s['y'])
            if s['drop_empty']:
                self.drop_empty_zone = (s['x'], s['y'])
            if s['charging_station']:
                self.charging_station_zones.append((s['x'], s['y']))

        self.initial_position = {}
        for agent in range(5):
            self.initial_position[agent] = self.charging_station_zones[0]
        for agent in range(5, 10):
            self.initial_position[agent] = self.charging_station_zones[1]

        possible_positions = {(s['x'], s['y']) for s in self.environment}
        self.transitions = {}
        for pos in possible_positions:
            x, y = pos
            self.transitions[pos] = {}


    def __update_repr(self):
        pass

    def step(self, actions):
        raise NotImplementedError


if __name__ == '__main__':
    dummy_map_file = '/docs/maps/dummy_map_2.json'
    with open(dummy_map_file) as f:
        env_map = json.load(f)

    env = ZalandoEnvironment(env_map, GlobalObservation)
    obs = env.reset()
    print(obs, env.malfunction)

    obs, r = env.step({0: 'E'})
    print(obs, r, env.malfunction)

    obs, r = env.step({0: 'N'})
    print(obs, r, env.malfunction)

    print('-----------')
    print(env.transitions)

    g = {key: list(set(values.values())) for key, values in env.transitions.items()}