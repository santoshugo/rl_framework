import numpy as np
from typing import Dict

from projects.dummy_project_2.utils import create_graph
from projects.dummy_project_2.agent import ZalandoAgent

ACTIONS = {'charge': -1, 'move': -2, 'pick': -3, 'drop': -4}
PICKUP_FULL = {6, 7, 8}
PICKUP_EMPTY = {0, 3}
CHARGING_STATION = {1, 5}
DROP_FULL = {2}
DROP_EMPTY = {4}

REWARDS = {'penalty': -np.inf, 'charge': 0.1, 'pickup': 10, 'dropdown': 100, 'move': -0.5}


class ZalandoEnvironment:

    def __init__(self, environment_map, agents: Dict[int, ZalandoAgent], initial_state: Dict[int, tuple], observation_obj, pickup_refill_probability):

        self.graph = create_graph(environment_map)

        self.agents = agents
        self.initial_state = initial_state

        self.observation = observation_obj(self)
        self.agents = agents

        for id, agent in self.agents.items():
            agent.set_available_options(self.graph)

        self.pickup_refill_probability = pickup_refill_probability

        self.pickup_carts = None
        self.charging_station_carts = {1: set(), 5: set()}

        self.no_resets = 0

    def reset(self):
        """
        Resets environment to initial state
        :return:
        """
        for key, value in self.initial_state.items():
            self.agents[key].state = value[0]
            self.agents[key].state_type = value[1]

            if value[0] == 1:
                self.charging_station_carts[1].add(key)
            elif value[0] == 5:
                self.charging_station_carts[5].add(key)

        self.pickup_carts = {pickup: 5 for pickup in self.pickup_refill_probability.keys()}

        self.no_resets += 1

        return self.observation.get_all()

    def step(self, options: dict):
        """
        Advances one time step on the environment. Returns env observation and (reward, option length) for agent
        on option finish.
        :param options:
        :return:
        """
        if self.no_resets == 0:
            raise Exception('Initial environment reset is required')

        # refill pickup nodes
        self.refill_pickup()

        reward = {}

        for agent_no, agent in self.agents.items():

            state, state_type = agent.state, agent.state_type

            # if no option is active raises an error
            if (agent.option is None or len(agent.action_sequence) == 0) and agent_no not in options.keys():
                raise Exception('No option is defined for agent {}'.format(agent_no))

            # if option is invalid, penalizes agent and nothing happens
            if agent_no in options.keys():
                option = options[agent_no]
                if option is None and option not in agent.get_available_options():
                    reward[agent_no] = (REWARDS['penalty'], 0)
                    continue
                else:
                    agent.set_option(option)

            action = agent.next_action()

            # if agent has no battery penalizes agent and returns it to a random charging station
            if action != -1:
                agent.decay()
                if agent.battery <= 0:
                    reward[agent_no] = (REWARDS['penalty'], 0)

                    station = 1 if len(self.charging_station_carts[1]) < 5 else 5
                    self.charging_station_carts[station].add(agent_no)
                    agent.set_state(station, 'node')

                    continue

            # agent charging
            if action == -1:
                self.charging_station_carts[state].add(agent_no)

                if len(self.charging_station_carts[state]) > 5:
                    reward[agent_no] = (REWARDS['penalty'], 0)
                else:
                    agent.charge()
                    if agent.charge == 1:
                        reward[agent_no] = (0, 0)
                    else:
                        reward[agent_no] = (REWARDS['charge'], 0)

            # agent picking up
            elif action == -3:
                # agent is penalized if no cart is present or is already carrying something
                if self.pickup_carts[state] == 0 or agent.carrying_full or agent.carrying_empty:
                    reward[agent_no] = (REWARDS['penalty'], 0)
                else:
                    self.pickup_carts[state] -= 1
                    reward[agent_no] = (REWARDS['pickup'], 0)

                    if state in PICKUP_FULL:
                        agent.carrying_full = True
                    else:
                        agent.carrying_empty = True

            # agent dropping
            elif action == -4:
                # agent is penalized if it is not carrying anything
                if not agent.carrying_full and not agent.carrying_empty:
                    reward[agent_no] = (REWARDS['penalty'], 0)
                else:
                    reward[agent_no] = (REWARDS['dropdown'], 0)
                    agent.carrying_full = False
                    agent.carrying_empty = False

            else:
                if agent.action_sequence is None:
                    reward[agent_no] = (REWARDS['move'], agent.option_len)
                    agent.state = action
                    agent.state_type = 'node'
                else:
                    agent.state = agent.option
                    agent.state_type = 'edge'

        return self.observation.get_all(), reward

    def refill_pickup(self):
        """
        Refills pickup spaces
        :return:
        """
        for pickup in self.pickup_carts.keys():
            no_vacancies = 5 - self.pickup_carts[pickup]
            additional_carts = 0

            for _ in range(no_vacancies):
                if np.random.random() < self.pickup_refill_probability[pickup]:
                    additional_carts += 1

            self.pickup_carts[pickup] += additional_carts


class ZalandoObservation:
    def __init__(self, environment):
        self.env = environment
        self.agents = environment.agents

    def get(self, agent):
        return agent.state, agent.state_type

    def get_all(self):
        observation = {}
        for agent_no, agent in self.agents.items():
            observation[agent_no] = self.get(agent)

        return observation
