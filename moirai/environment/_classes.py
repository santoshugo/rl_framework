from rl_framework.utils.typing import AgentID, AgentObs, AgentAction, AgentState, List


class AbstractAgent:
    def __init__(self, id: AgentID, environment):
        self.id = id
        self.environment = environment

    def get_available_actions(self) -> List[AgentAction]:
        raise NotImplementedError

    def set_state(self, state: AgentState):
        raise NotImplementedError

    def step(self, action: AgentAction) -> AgentObs:
        raise NotImplementedError

