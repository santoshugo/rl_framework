import numpy as np
import networkx as nx


ACTIONS = {'charge': -1, 'move': -2, 'pick': -3, 'drop': -4}
PICKUP_FULL = {6, 7, 8}
PICKUP_EMPTY = {0, 3}
CHARGING_STATION = {1, 5}
DROP_FULL = {2}
DROP_EMPTY = {4}

REWARDS = {'penalty': -np.inf, 'charge': 0.1, 'pickup': 10, 'dropdown': 100, 'move': -0.5}


class ZalandoEnvironment:
    def __init__(self, environment_map, agents, initial_state, observation_obj, pickup_refill_probability):

        self.graph = Graph(environment_map).graph

        self.agents = agents
        self.initial_state = initial_state

        self.observation = observation_obj(self)
        self.agents = agents

        self.pickup_refill_probability = pickup_refill_probability

        self.pickup_carts = None
        self.charging_station_carts = {1: set(), 5: set()}

        self.no_resets = 0

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


class Graph:
    def __init__(self, map):
        self.nodes = map['nodes']
        self.edges = map['edges']

        self.directed = map['directed']

        if self.directed:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

        self._create_graph()

    def _create_graph(self):

        for node in self.nodes:
            node_id = node.pop('id')
            self.graph.add_node(node_id, **node)
            self.graph.nodes[node_id]['agent'] = None

        for edge in self.edges:
            node_1 = edge.pop('node_1')
            node_2 = edge.pop('node_2')

            distance = self._compute_distance(node_1, node_2)

            self.graph.add_edge(node_1, node_2, agent=None, distance=distance, **edge)

    def _compute_distance(self, node_1, node_2):

        x_diff = self.graph.nodes[node_1]['x'] - self.graph.nodes[node_2]['x']
        y_diff = self.graph.nodes[node_1]['y'] - self.graph.nodes[node_2]['y']

        return np.sqrt(np.square(x_diff) + np.square(y_diff))


class ZalandoObservation:
    def __init__(self, environment):
        self.env = environment

    def get(self, agent):
        return agent.state, agent.state_type, agent.objective_node, agent.battery

    def get_all(self):
        observation = {}
        for agent_no, agent in self.env.agents.items():
            observation[agent_no] = self.get(agent)

        return observation