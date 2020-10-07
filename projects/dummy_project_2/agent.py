class ZalandoAgent:
    def __init__(self, id, speed, battery_decay_function, battery_charge_function):
        self.id = id

        self.state = None
        self.state_type = None

        self.option = None

        self.battery = 1
        self.battery_decay_function = battery_decay_function
        self.battery_charge_function = battery_charge_function
        self.speed = speed

        self.carrying_empty = False
        self.carrying_full = False

    def charge(self):
        self.battery = min(1, self.battery_charge_function(self.battery))

    def decay(self):
        self.battery = max(0, self.battery_decay_function(self.battery))

    def set_state(self, state, state_type):
        self.state = state
        self.state_type = state_type

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

    def get_available_options(self):
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