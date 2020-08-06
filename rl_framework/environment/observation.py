from rl_framework.environment.utils import dfs


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


class LocalSearchObservation(AbstractObservation):
    """
    Returns shortest path to destination for each agent
    """
    def __init__(self, environment):
        self.environment = environment
        self.__precompute()

    def __precompute(self):
        # check shortes path from all points to all other points
        shortest_path = dfs(self.environment.transitions)

    def get(self, agent):
        pass

    def get_all(self):
        return {agent: self.get(agent) for agent in self.environment.agents}


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
