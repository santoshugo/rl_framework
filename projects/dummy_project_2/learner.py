from random import choice


class ZalandoLearner:
    def __init__(self, agents):
        self.agents = agents

    def _random_option(self, agent):
        return choice(tuple(agent.get_available_options()))

    def get_random_options(self):
        random_options = {}
        for agent_no, agent in self.agents.items():
            if agent.option is None:
                random_options[agent_no] = self._random_option(agent)

        return random_options
