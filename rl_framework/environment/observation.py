class AbstractObservation:
    """
    Abstract class that implements base observation logic and speeds development of project-level observations
    """
    def get(self, agent):
        """
        Returns observation for a single agent
        :return:
        """
        raise NotImplementedError

    def get_all(self):
        """
        Returns observation for all agents
        :return:
        """
        raise NotImplementedError


class GlobalObservation(AbstractObservation):
    """
    Simple global observation class. Returns current position of all agents in environment,
    """
    def __init__(self, environment):
        self.environment = environment

    def get(self, agent):
        return self.environment.state[agent]

    def get_all(self):
        return {agent: self.get(agent) for agent in self.environment.agents}
