import json
import numpy as np

from rl_framework.environment.environment import GraphEnvironment
from rl_framework.environment.observation import GlobalObservation, AbstractObservation
from rl_framework.solver.agent import AbstractAgent


class ZalandoEnvironment(GraphEnvironment):
    ACTIONS = {'charge': 0, 'move': 1}

    def __init__(self, environment_map, observation_obj, initial_state, pickup_refill_probability):
        super().__init__(environment_map, observation_obj, initial_state, malfunction_prob=0, malfunction_len=0)

        self.pickup_refill_probability = pickup_refill_probability
        self.pickup_carts = {pickup: 5 for pickup in pickup_refill_probability.keys()}

    def step(self, actions):
        if self.no_resets == 0:
            raise Exception('Initial environment reset is required')

        reward = {}

        for agent, action in actions.items():
            pass
            # if action == pass and agent at charging station

            # if action == edge and agent at node

            # if action == pickup / dropdown and agent at node

            # if action == move and agent at edge

    def refill_pickup(self):
        for pickup in self.pickup_carts.keys():
            no_vacancies = 5 - self.pickup_carts[pickup]
            additional_carts = 0

            for _ in range(no_vacancies):
                if np.random.random() < self.pickup_refill_probability[pickup]:
                    additional_carts += 1

            self.pickup_carts[pickup] += additional_carts

    def __update_repr(self):
        pass


class ZalandoObservation(AbstractObservation):
    def get(self, agent):
        pass

    def get_all(self):
        pass


class ZalandoAgent(AbstractAgent):
    def __init__(self, id, initial_state, speed, battery_decay_function):
        super().__init__(id, initial_state)
        self.speed = speed
        self.battery_decay_function = battery_decay_function


if __name__ == '__main__':
    dummy_map_file = 'C:\\Users\\santosh\\PycharmProjects\\rl_framework\\docs\\maps\\dummy_map_2.json'
    with open(dummy_map_file) as f:
        env_map = json.load(f)

    # refills each cart with probability 1 / n
    pickup_refill = {'pickup full 1': 20, 'pickup full 2': 20, 'pickup full 3': 20, 'pickup empty 1': 20, 'pickup empty 2': 20}

    initial_nodes = {0: 1,
                     1: 1,
                     2: 1,
                     3: 1,
                     4: 1,
                     5: 5,
                     6: 5,
                     7: 5,
                     8: 5,
                     9: 5}

    env = ZalandoEnvironment(env_map, GlobalObservation, initial_nodes, pickup_refill)
    obs = env.reset()

    agents = {agent: ZalandoAgent(agent) for agent in env.agents}

    print(type(env.graph.nodes[0]))
    print(type(env.graph.edges[0, 1]))

    obs, reward = env.step()

    print(obs, env.malfunction)
