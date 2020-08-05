class Agent:
    """
    Single agent that operates on the environment.
    - Knows which actions/options are available in each state;
    - Decomposes each option in its constituent atomic actions;
    - Receives observation from environment;
    - Applies atomic action to environment;
    - Communicates with manager.
    """
    def __init__(self):
        raise NotImplementedError
