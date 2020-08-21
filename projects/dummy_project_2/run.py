import json
import numpy as np

from rl_framework.environment.environment import GraphEnvironment
from rl_framework.environment.observation import GlobalObservation, AbstractObservation
from rl_framework.solver.actions import Option


class ZalandoEnvironment(GraphEnvironment):
    def __init__(self, environment_map, observation_obj, initial_state, pickup_refill_probability):
        super().__init__(environment_map, observation_obj, initial_state, malfunction_prob=0, malfunction_len=0)

        self.pickup_refill_probability = pickup_refill_probability
        self.pickup_carts = {pickup: 5 for pickup in pickup_refill_probability.keys()}

    def step(self, actions):
        pass

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


class ZalandoOption(Option):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    dummy_map_file = '/home/hugo/PycharmProjects/rl_framework/docs/maps/dummy_map_2.json'
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

    print(env.graph.nodes.data())
    print(env.graph.edges.data())

    print(obs, env.malfunction)
