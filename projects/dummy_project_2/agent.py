import numpy as np
from copy import deepcopy


class ZalandoAgent:
    def __init__(self, id, speed, battery_decay_function, battery_charge_function):
        self.id = id

        self.state = None
        self.state_type = None

        self.option = None
        self.action_sequence = None
        self.option_len = None

        self.available_options = None
        self.graph = None

        self.battery = 1
        self.battery_decay_function = battery_decay_function
        self.battery_charge_function = battery_charge_function
        self.speed = speed

        self.carrying_empty = False
        self.carrying_full = False

    def charge(self):
        """
        Charges battery
        :return:
        """
        self.battery = min(1, self.battery_charge_function(self.battery))

    def decay(self):
        """
        Decays battery value
        :return:
        """
        self.battery = max(0, self.battery_decay_function(self.battery))

    def set_available_options(self, graph):
        """
        Sets all available options as well as the option policy for agent
        :param graph:
        :return:
        """
        self.graph = graph
        options = {-1: [-1], -3: [-3], -4: [-4]}
        for start_node in graph.nodes:
            for destination_node, attr in graph.adj[start_node].items():
                distance = attr['distance']
                no_time_steps = np.ceil(distance / self.speed)

                options[(start_node, destination_node)] = int(no_time_steps) * [destination_node]

        self.available_options = options

    def set_option(self, option):
        """
        Sets current option
        :param option:
        :return:
        """
        self.option = option
        self.action_sequence = deepcopy(self.available_options[option])
        self.option_len = len(self.action_sequence)

    def set_state(self, state, state_type):
        """
        Sets current state
        :param state:
        :param state_type:
        :return:
        """
        self.state = state
        self.state_type = state_type

    def next_action(self):
        action = self.action_sequence.pop(0)
        if len(self.action_sequence) == 0:
            self.option = None
            self.action_sequence = None
            self.option_len = None

        return action

    def get_available_options(self):
        """
        Returns available options for the current agent state
        :return:
        """
        if self.option is not None:
            return set()

        options = set()

        for destination_node in self.graph.adj[self.state].keys():
            options.add((self.state, destination_node))

        if self.state in [0, 3, 6, 7, 8]:
            options.add(-3)
        elif self.state in [1, 5]:
            options.add(-1)
        elif self.state in [2, 4]:
            options.add(-4)

        if self.carrying_empty or self.carrying_full:
            options.discard(-3)
        if not self.carrying_empty:
            options.discard(-4)
        if not self.carrying_full:
            options.discard(-4)

        return options

    def get_available_actions(self):
        """
        Returns available atomic actions for the current agent state
        :return:
        """
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