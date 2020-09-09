import json
import numpy as np

from rl_framework.environment.environment import GraphEnvironment
from rl_framework.environment.observation import GlobalObservation, AbstractObservation
from rl_framework.solver.agent import AbstractAgent


class ZalandoEnvironment(GraphEnvironment):
    ACTIONS = {'charge': -1, 'move': -2, 'pick': -3, 'drop': -4}

    def __init__(self, environment_map, agents, observation_obj, initial_state, pickup_refill_probability):
        super().__init__(environment_map, observation_obj, initial_state, malfunction_prob=0, malfunction_len=0)

        self.agents = agents

        self.pickup_refill_probability = pickup_refill_probability
        self.pickup_carts = {pickup: 5 for pickup in pickup_refill_probability.keys()}

        self.charging_station_carts = {1: {0, 1, 2, 3, 4}, 5: {5, 6, 7, 8, 9}}

    def step(self, actions):
        if self.no_resets == 0:
            raise Exception('Initial environment reset is required')

        reward = {}

        for agent_no, action in actions.items():
            agent = self.agents[agent_no]

            # if agent is at a charging station and wants to charge
            if action == -1 and self.state[agent_no]['description'] == "charging station":
                self.charging_station_carts[self.state[agent_no]].add(agent_no)

                if len(self.charging_station_carts[self.state[agent_no]]) > 5:
                    reward[agent_no] = -np.inf
                else:
                    reward[agent_no] = 1

                agent.charge()

            # if agent at a pickup station and wants to pick
            elif action == -3 and 'pickup' in self.state[agent_no]['description']:
                pass

            # if agent at a drop station and wants to pick
            elif action == -4 and 'drop' in self.state[agent_no]['description']:
                pass

            # if agent at a node and should start to move to another node
            elif action not in [-1, -2]:
                pass
                # error catching

            # if agent is moving in an edge
            elif action == -2:
                pass
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

    def check_charging_capacity(self):
        pass

    def __update_repr(self):
        pass


class ZalandoObservation(AbstractObservation):
    def get(self, agent):
        pass

    def get_all(self):
        pass


class ZalandoAgent(AbstractAgent):
    def __init__(self, id, initial_state, speed, battery_decay_function, battery_charge_function):
        super().__init__(id, initial_state)
        self.speed = speed
        self.battery_decay_function = battery_decay_function
        self.battery_charge_function = battery_charge_function
        self.battery = 1

    def charge(self):
        self.battery = min(1, self.battery_charge_function(self.battery))

    def decay(self):
        self.battery = max(0, self.battery_decay_function(self.battery))


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

    agents = {agent_no: ZalandoAgent(agent_no,
                                     initial_nodes[agent_no],
                                     0.5,
                                     lambda x: x - 100 / (90 * 60),
                                     lambda x: x + 100 / (180 * 60)
                                     ) for agent_no in range(10)}

    env = ZalandoEnvironment(env_map, agents, GlobalObservation, initial_nodes, pickup_refill)
    obs = env.reset()

    obs, reward = env.step()

    print(obs, env.malfunction)
