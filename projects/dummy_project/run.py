import json
from copy import deepcopy
import numpy as np

from rl_framework.environment.environment import FiniteGridEnvironment
from rl_framework.environment.observation import GlobalObservation, AbstractObservation
from rl_framework.solver.agent import AbstractAgent
from rl_framework.environment.utils import bfs


class DummyEnvironment(FiniteGridEnvironment):
    def __init__(self, environment_map, observation_obj, malfunction_prob=0.05, malfunction_len=1):
        super().__init__(environment_map, observation_obj, malfunction_prob, malfunction_len)

        self.initial_representation = np.array([[' X ' for _ in range(self.width)] for _ in range(self.height)])
        for x, y in self.transitions.keys():
            self.initial_representation[self.height - 1 - y][x] = '   '

    def step(self, actions):
        if self.no_resets == 0:
            raise Exception('Initial environment reset is required')

        reward = {}

        for agent, action in actions.items():
            # if agent reached destination, reward is 0 and agent does not move
            if self.done[agent]:
                reward[agent] = 0
                continue

            # if agent is not malfunctioning, there is a probability of malfunction
            if np.random.random() < self.malfunction_prob and not self.malfunction[agent]:
                self.__break_agent(agent)
            # if is malfunctioning, update it
            if self.malfunction[agent]:
                reward[agent] = -1
                self.__update_broken_agent(agent)
                continue

            # otherwise runs as normal
            if not self.done[agent]:
                self.state[agent] = self.transitions[self.state[agent]][self.ACTIONS[action]]

            if self.state[agent] == self.terminal_position[agent] and not self.done[agent]:
                reward[agent] = 1
            elif self.done[agent]:
                reward[agent] = 0
            else:
                reward[agent] = -1

        self.__update_repr()

        return self.observation.get_all(), reward

    def __update_repr(self):
        self.representation = deepcopy(self.initial_representation)
        for agent, (x, y) in self.state.items():
            if self.representation[self.height - 1 - y][x] == '   ':
                self.representation[self.height - 1 - y][x] = ' {} '.format(agent)
            else:
                self.representation[self.height - 1 - y][x] = ' M '.format(agent)

    def __repr__(self):
        return str(self.representation)


class ShortestPathObservation(AbstractObservation):
    def __init__(self, environment):
        self.environment = environment
        self.graph = {state: list(set(action_s_prime.values())) for state, action_s_prime in self.environment.transitions.items()}

        self.shortest_path = {agent: bfs(self.graph, self.environment.terminal_position[agent]) for agent in self.environment.agents}

    def get(self, agent):
        state = self.environment.state[agent]
        shortest_path = [state]
        while True:
            state = self.shortest_path[agent][state]
            shortest_path.append(state)

            if state == self.environment.terminal_position[agent]:
                break

        return shortest_path

    def get_all(self):
        return {agent: self.get(agent) for agent in self.environment.agents}


class GridAgent(AbstractAgent):
    def __init__(self, id, initial_state):
        super().__init__(id, initial_state)
        self.option = None

    def option_to_action(self):
        raise NotImplementedError

    def get_valid_options(self, state):
       pass

    def set_option(self, option):
        self.option = option


if __name__ == '__main__':
    dummy_map_file = '/docs/maps/dummy_map.json'
    with open(dummy_map_file) as f:
        env_map = json.load(f)

    env = DummyEnvironment(env_map, ShortestPathObservation)
    obs = env.reset()
    print(obs, env.malfunction)

    obs, r = env.step({0: 'E'})
    print(obs, r, env.malfunction)

    obs, r = env.step({0: 'N'})
    print(obs, r, env.malfunction)

    print('-----------')
    print(env.transitions)

    g = {key: list(set(values.values())) for key, values in env.transitions.items()}
