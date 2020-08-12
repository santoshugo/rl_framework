class Manager:
    """
    Orchestrates multiple agents to find optimal global policy.
    - Receives information from agent;
    - Learns optimal policy (?);
    - Detects any conflict caused by agent-level policy;
    - Breaks options;
    """
    def __init__(self, env, agent_class, options_class):
        self.agents = {agent: agent_class() for agent in range(env.agents)}
        self.options = {agent: options_class(agent) for agent in range(env.agents)}

        self.__build_initial_value_function()

    def __build_initial_value_function(self):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

    def act(self):
        raise NotImplementedError
