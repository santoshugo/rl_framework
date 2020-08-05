from rl_framework.environment.environment import AbstractEnvironment
from rl_framework.environment.observation import AbstractObservation


class DummyObs(AbstractObservation):
    def __init__(self):
        super().__init__()

    def get(self):
        pass


class DummyEnv(AbstractEnvironment):
    def __init__(self):
        super().__init__()

    def step(self):
        pass
