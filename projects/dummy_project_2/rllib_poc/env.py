from gym.spaces import Discrete
import json
import numpy as np

from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from projects.dummy_project_2.agent import ZalandoAgent
from projects.dummy_project_2.environment import ZalandoObservation
from projects.dummy_project_2.utils import create_graph

MAP_PATH = 'C:\\Users\\santosh\\PycharmProjects\\rl_framework\\docs\\maps\\dummy_map_2.json'
with open(MAP_PATH) as f:
    env_map = json.load(f)


def battery_decay_function(x):
    return x - 100 / (90 * 60)


def battery_charge_function(x):
    return x + 100 / (180 * 60)


initial_state = {0: (1, 'node'),
                 1: (1, 'node'),
                 # 2: (1, 'node'),
                 # 3: (1, 'node'),
                 # 4: (1, 'node'),
                 # 5: (5, 'node'),
                 # 6: (5, 'node'),
                 # 7: (5, 'node'),
                 # 8: (5, 'node'),
                 # 9: (5, 'node')
                 }


class ZalandoEnv(MultiAgentEnv):
    action_space = Discrete(10)  # change this

    def __init__(self, env_config):
        self.state = None
        self.initial_state = initial_state

        self.graph = create_graph(env_map)
        self.observation = ZalandoObservation(self)
        self.agents = {agent_no: ZalandoAgent(agent_no, 0.5, battery_decay_function, battery_charge_function) for agent_no in range(len(initial_state))}

        for id, agent in self.agents.items():
            agent.set_available_options(self.graph)

        self.pickup_refill_probability = {0: 20, 3: 20, 6: 20, 7: 20, 8: 20}

        self.pickup_carts = None
        self.charging_station_carts = {1: set(), 5: set()}

    def reset(self):
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

    def step(self, action_dict):
        """
        Advances one time step on the environment. Returns env observation and (reward, option length) for agent
        on option finish.
        """

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
                    reward[agent_no] = (REWARDS['charge_penalty'], 0)

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