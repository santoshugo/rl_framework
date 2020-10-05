class Manager:
    """
    Orchestrates multiple agents to find optimal global policy.
    - Receives information from agent;
    - Learns optimal policy (?);
    - Detects any conflict caused by agent-level policy;
    - Breaks options;
    """
    def __init__(self, environment, agents, model):
        self.environment = environment
        self.agents = agents
        self.model = model

    def learn(self):
        raise NotImplementedError

    def get_actions(self):
        raise NotImplementedError


class ApproximationModel:
    pass