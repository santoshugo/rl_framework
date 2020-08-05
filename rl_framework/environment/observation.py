class AbstractObservation:
    """
    Abstract class that implements base observation logic and speeds development of project-level observations
    """
    def __init__(self):
        raise NotImplementedError

    def get(self):
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
    def __init__(self):
        super().__init__()

    def get(self):
        pass

    def get_all(self):
        pass
