from copy import deepcopy
import numpy as np
import networkx as nx


class AbstractEnvironment:
    """
    Template class with the required functions that should be implemented on the environment class
    """
    def reset(self):
        """
        Resets environment to initial state.
        """
        raise NotImplementedError

    def step(self, actions):
        """
        Should update internal environment structure based on the actions it receives, and return an observation and
        a reward (optionally also more info)
        :return: observation, reward, done (optional), info (optional)
        """
        raise NotImplementedError


class FiniteGridEnvironment(AbstractEnvironment):
    """
    Abstract class that implements gridworld-like environment logic and speeds development of project-level environments.
    Inputs may be defined either programmatically or graphically.
    """
    ACTIONS = {'N': 0, 'E': 1, 'S': 2, 'W': 3, 'P': 4}  # North | East | South | West | Do nothing

    def __init__(self, environment_map, observation_obj, malfunction_prob, malfunction_len):
        """
        Should take in an observation type (to return on step call) and a environment map
        """
        self.env_type = 'finite_grid'

        self.width = environment_map['width']
        self.height = environment_map['height']
        self.no_agents = environment_map['no_agents']
        self.agents = {agent for agent in range(self.no_agents)}

        self.environment = environment_map['environment']
        self.__build()

        self.observation = observation_obj(self)

        self.state = None
        self.no_resets = 0

        self.done = {agent: False for agent in self.agents}
        self.done['__all__'] = False

        self.malfunction = {agent: False for agent in self.agents}
        self.__time_until_restore = {agent: 0 for agent in self.agents}

        self.malfunction_prob = malfunction_prob
        self.malfunction_len = malfunction_len

    def __build(self):
        """
        Sets initial agent position, creates transition map for grid environment and sets initial grid repr
        :return:
        """
        self.initial_position = {}
        self.terminal_position = {}

        for s in self.environment:
            if s['start'] is not None:
                self.initial_position[s['start']] = (s['x'], s['y'])

            if s['destination'] is not None:
                self.terminal_position[s['destination']] = (s['x'], s['y'])

        possible_positions = {(s['x'], s['y']) for s in self.environment}
        self.transitions = {}
        for pos in possible_positions:
            x, y = pos
            self.transitions[pos] = {}

            self.transitions[pos][0] = (x, y + 1) if (x, y + 1) in possible_positions else pos  # move north
            self.transitions[pos][1] = (x + 1, y) if (x + 1, y) in possible_positions else pos  # move east
            self.transitions[pos][2] = (x, y - 1) if (x, y - 1) in possible_positions else pos  # move south
            self.transitions[pos][3] = (x - 1, y) if (x - 1, y) in possible_positions else pos  # move west
            self.transitions[pos][4] = pos  # don't move

    def reset(self):
        self.state = deepcopy(self.initial_position)
        self.no_resets += 1

        self.done = {agent: False for agent in self.agents}
        self.done['__all__'] = False

        return self.observation.get_all()

    def __break_agent(self, agent):
        self.malfunction[agent] = True
        self.malfunction_len[agent] = np.random.poisson(self.malfunction_len) + 1

    def __update_broken_agent(self, agent):
        self.malfunction_len[agent] -= 1

        if self.malfunction_len[agent] <= 0:
            self.malfunction[agent] = False
            self.malfunction_len[agent] = 0

    def __update_repr(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError


class InfiniteGridEnvironment(AbstractEnvironment):
    """
    Abstract class that implements infinite gridworld-like environment (repeating tasks)logic and speeds development of project-level environments.
    Inputs may be defined either programmatically or graphically.
    """
    ACTIONS = {'N': 0, 'E': 1, 'S': 2, 'W': 3, 'P': 4}  # North | East | South | West | Do nothing

    def __init__(self, environment_map, observation_obj, malfunction_prob, malfunction_len):
        """
        Should take in an observation type (to return on step call) and a environment map
        """
        self.env_type = 'infinite_grid'

        self.width = environment_map['width']
        self.height = environment_map['height']
        self.no_agents = environment_map['no_agents']
        self.agents = {agent for agent in range(self.no_agents)}

        self.environment = environment_map['environment']
        self.__build()

        self.observation = observation_obj(self)

        self.state = None
        self.no_resets = 0

        self.malfunction = {agent: False for agent in self.agents}
        self.__time_until_restore = {agent: 0 for agent in self.agents}

        self.malfunction_prob = malfunction_prob
        self.malfunction_len = malfunction_len

    def __build(self):
        """
        Sets initial agent position, creates transition map for grid environment and sets initial grid repr
        :return:
        """
        self.initial_position = {}
        for s in self.environment:
            if s['start'] is not None:
                self.initial_position[s['start']] = (s['x'], s['y'])

        possible_positions = {(s['x'], s['y']) for s in self.environment}
        self.transitions = {}
        for pos in possible_positions:
            x, y = pos
            self.transitions[pos] = {}

            self.transitions[pos][0] = (x, y + 1) if (x, y + 1) in possible_positions else pos  # move north
            self.transitions[pos][1] = (x + 1, y) if (x + 1, y) in possible_positions else pos  # move east
            self.transitions[pos][2] = (x, y - 1) if (x, y - 1) in possible_positions else pos  # move south
            self.transitions[pos][3] = (x - 1, y) if (x - 1, y) in possible_positions else pos  # move west
            self.transitions[pos][4] = pos  # don't move

    def reset(self):
        self.state = deepcopy(self.initial_position)
        self.no_resets += 1

        return self.observation.get_all()

    def __break_agent(self, agent):
        self.malfunction[agent] = True
        self.malfunction_len[agent] = np.random.poisson(self.malfunction_len) + 1

    def __update_broken_agent(self, agent):
        self.malfunction_len[agent] -= 1

        if self.malfunction_len[agent] <= 0:
            self.malfunction[agent] = False
            self.malfunction_len[agent] = 0

    def __update_repr(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError


class FiniteGraphEnvironment(AbstractEnvironment):
    """
    Abstract class that implements graph-like environment logic and speeds development of project-level environments.
    Inputs may be defined either programmatically or graphically.
    """

    def __init__(self, environment_map, observation_obj):
        self.env_type = 'finite_graph'

        self.nodes = environment_map['nodes']
        self.edges = environment_map['edges']

        self.directed = environment_map['directed']
        self.no_agents = environment_map['no_agents']

    def reset(self):
        pass

    def step(self, actions):
        pass


class InfiniteGraphEnvironment(AbstractEnvironment):
    """
    Abstract class that implements graph-like environment logic and speeds development of project-level environments.
    Inputs may be defined either programmatically or graphically.
    """

    def __init__(self, environment_map, observation_obj):
        self.env_type = 'infinite_grid'

        self.nodes = environment_map['nodes']
        self.edges = environment_map['edges']

        self.directed = environment_map['directed']
        self.no_agents = environment_map['no_agents']

        self.graph = nx.Graph()
        self._create_graph()

    def _create_graph(self):
        import matplotlib.pyplot as plt

        for node in self.nodes:
            node_id = node.pop('id')
            self.graph.add_node(node_id, **node)

        for edge in self.edges:
            node_1 = edge.pop('node_1')
            node_2 = edge.pop('node_2')

            self.graph.add_edge(node_1, node_2, **edge)


    def reset(self):
        pass

    def step(self, actions):
        pass
