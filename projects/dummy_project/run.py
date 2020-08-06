import json
from copy import deepcopy
import numpy as np

from rl_framework.environment.environment import GridEnvironment
from rl_framework.environment.observation import GlobalObservation


class DummyEnvironment(GridEnvironment):
    def __init__(self, environment_map, observation_obj):
        super().__init__(environment_map, observation_obj)

        self.initial_representation = np.array([[' X ' for _ in range(self.width)] for _ in range(self.height)])
        for x, y in self.transitions.keys():
            self.initial_representation[self.height - 1 - y][x] = '   '

        self.destination = {}
        for s in self.environment:
            if s['destination'] is not None:
                self.destination[s['destination']] = (s['x'], s['y'])

        self.done = {agent: False for agent in self.agents}
        self.done['__all__'] = False

    def step(self, actions):
        if self.no_resets == 0:
            raise Exception('Initial environment reset is required')

        reward = {}

        for agent, action in actions.items():
            if not self.done[agent]:
                self.state[agent] = self.transitions[self.state[agent]][self.ACTIONS[action]]

            if self.state[agent] == self.destination[agent] and not self.done[agent]:
                reward[agent] = 1
            elif self.done[agent]:
                reward[agent] = 0
            else:
                reward[agent] = -1

        self.__update_repr()

        return self.observation.get_all(), reward

    def __break_agent(self):
        pass

    def __restore_agent(self):
        pass

    def __update_repr(self):
        self.representation = deepcopy(self.initial_representation)
        for agent, (x, y) in self.state.items():
            if self.representation[self.height - 1 - y][x] == '   ':
                self.representation[self.height - 1 - y][x] = ' {} '.format(agent)
            else:
                self.representation[self.height - 1 - y][x] = ' M '.format(agent)

    def __repr__(self):
        return str(self.representation)


if __name__ == '__main__':
    dummy_map_file = 'C:\\Users\\santosh\\PycharmProjects\\rl_framework\\docs\\dummy_map.json'
    with open(dummy_map_file) as f:
        env_map = json.load(f)

    env = DummyEnvironment(env_map, GlobalObservation)
    obs = env.reset()
    print(obs)

    obs, r = env.step({0: 'E'})
    print(obs, r)

    obs, r = env.step({0: 'N'})
    print(obs, r)
