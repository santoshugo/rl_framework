import json
import numpy as np

from rl_framework.environment.environment import GraphEnvironment
from rl_framework.environment.observation import GlobalObservation, AbstractObservation
from rl_framework.solver.agent import AbstractAgent

ACTIONS = {'charge': -1, 'move': -2, 'pick': -3, 'drop': -4}
PICKUP_FULL = {6, 7, 8}
PICKUP_EMPTY = {0, 3}
CHARGING_STATION = {1, 5}
DROP_FULL = {2}
DROP_EMPTY = {4}

REWARDS = {'penalty': -np.inf, 'charge': 0.1, 'pickup': 10, 'dropdown': 100, 'move': -0.5}


class ZalandoEnvironment(GraphEnvironment):
    def __init__(self, environment_map, agents, observation_obj, initial_state, pickup_refill_probability):
        super().__init__(environment_map, observation_obj, initial_state, malfunction_prob=0, malfunction_len=0)

        self.observation = observation_obj(self)
        self.agents = agents

        self.pickup_refill_probability = pickup_refill_probability
        self.pickup_carts = {pickup: 5 for pickup in pickup_refill_probability.keys()}

        self.charging_station_carts = {1: {0, 1, 2, 3, 4}, 5: {5, 6, 7, 8, 9}}

    def step(self, actions):
        if self.no_resets == 0:
            raise Exception('Initial environment reset is required')

        self.refill_pickup()

        reward = {}

        for agent_no, action in actions.items():
            agent = self.agents[agent_no]
            state = agent.state
            state_type = agent.state_type

            if action != -1:
                agent.decay()
                if agent.battery <= 0:
                    reward[agent_no] = REWARDS['penalty']
                    continue

            if action not in agent.get_available_actions():
                reward[agent_no] = REWARDS['penalty']

            # agent charging
            elif action == -1:
                self.charging_station_carts[state].add(agent_no)
                import pandas as pd
                pd.Series().rank()
                if len(self.charging_station_carts[state]) > 5:
                    reward[agent_no] = REWARDS['penalty']
                else:
                    agent.charge()
                    if agent.charge == 1:
                        reward[agent_no] = 0
                    else:
                        reward[agent_no] = REWARDS['charge']

            # agent picking up
            elif action == -3:
                if self.pickup_carts[state] == 0:
                    reward[agent_no] = REWARDS['penalty']
                else:
                    self.pickup_carts[state] -= 1
                    reward[agent_no] = REWARDS['pickup']
                    if state in PICKUP_FULL:
                        agent.carrying_full = True
                    else:
                        agent.carrying_empty = True

            # agent dropping
            elif action == -4:
                reward[agent_no] = REWARDS['dropdown']
                agent.carrying_full = False
                agent.carrying_empty = False

            # moves to node
            elif action >= 0:
                reward[agent_no] = REWARDS['move']

                agent.set_node(action, self.graph.edges[state, action]['distance'])
                agent.add_distance()

            # moves in edge
            elif action == -2:
                reward[agent_no] = REWARDS['move']
                agent.add_distance()

                if agent.distance_to_node <= 0:
                    agent.reset_node()

        return self.observation.get_all(), reward

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
    def __init__(self, environment):
        self.env = environment

    def get(self, agent):
        return agent.state, agent.state_type, agent.objective_node, agent.battery

    def get_all(self):
        observation = {}
        for agent_no, agent in self.env.agents.items():
            observation[agent_no] = self.get(agent)

        return observation


class ZalandoAgent(AbstractAgent):
    def __init__(self, id, initial_state, speed, battery_decay_function, battery_charge_function):
        super().__init__(id, initial_state)
        self.state = initial_state
        self.state_type = 'node'
        self.objective_node = None
        self.distance_to_node = None
        self.speed = speed
        self.battery_decay_function = battery_decay_function
        self.battery_charge_function = battery_charge_function
        self.battery = 1

        self.carrying_empty = False
        self.carrying_full = False

    def charge(self):
        self.battery = min(1, self.battery_charge_function(self.battery))

    def decay(self):
        self.battery = max(0, self.battery_decay_function(self.battery))

    def set_node(self, node, distance):
        self.state_type = 'edge'
        self.objective_node = node

        self.distance_to_node = distance

    def reset_node(self):
        self.state_type = 'node'
        self.state = self.objective_node
        self.distance_to_node, self.objective_node = None, None

    def add_distance(self):
        self.distance_to_node -= self.speed

    def get_available_actions(self):
        # TODO complete
        if self.state_type == 'edge':
            return {-2}

        actions = {}
        if self.state == 0:
            actions = {-3, 1}
        elif self.state == 1:
            actions = {-1, 0, 2}
        elif self.state == 2:
            actions = {-4, 1, 3}
        elif self.state == 3:
            actions = {-3, 4}
        elif self.state == 4:
            actions = {-4, 5, 6}
        elif self.state == 5:
            actions = {-1, 4}
        elif self.state == 6:
            actions = {-3, 7}
        elif self.state == 7:
            actions = {-3, 8}
        elif self.state == 8:
            actions = {-3, 2}

        if self.carrying_empty or self.carrying_full:
            actions.discard(-3)
        if not self.carrying_empty:
            actions.discard(-4)
        if not self.carrying_full:
            actions.discard(-4)

        return actions


if __name__ == '__main__':
    dummy_map_file = 'C:\\Users\\santosh\\PycharmProjects\\rl_framework\\docs\\maps\\dummy_map_2.json'
    with open(dummy_map_file) as f:
        env_map = json.load(f)

    # refills each cart with probability 1 / n
    pickup_refill = {0: 20, 3: 20, 6: 20, 7: 20, 8: 20}

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

    env = ZalandoEnvironment(env_map, agents, ZalandoObservation, initial_nodes, pickup_refill)
    obs = env.reset()
    print(obs)

    actions = {0: 0, 1: 2, 3: -1, 4: -1, 5: 4, 6: -1, 7: -1, 8: -1, 9: -1}

    obs, reward = env.step(actions)

    print(obs)
    print(reward)
