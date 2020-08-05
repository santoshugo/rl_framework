class AbstractEnvironment:
    """
    Abstract class that implements base environment logic and speeds development of project-level environments.
    Inputs may be defined either programmatically or graphically.
    """
    def __init__(self):
        """
        Should take in an observation type (to return on step call) and a environment map
        """
        raise NotImplementedError

    def step(self):
        """
        Should update internal environment structure based on the actions it receives, and return an observation and
        a reward (optionally also more info)
        :return: observation
        """
        raise NotImplementedError

    def loads_json(self):
        """
        Loads environment information from JSON file
        """
        raise NotImplementedError
