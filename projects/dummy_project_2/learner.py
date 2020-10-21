from random import choice

from tensorflow import keras
from tensorflow.keras import layers

OPTIMIZER = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)


class ZalandoLearner:
    def __init__(self, agents, input_size, num_options):
        self.agents = agents

        self.model_network = create_approximation_model(input_size)
        self.target_network = create_approximation_model(input_size)

    def _random_option(self, agent):
        return choice(tuple(agent.get_available_options()))

    def get_random_options(self):
        random_options = {}
        for agent_no, agent in self.agents.items():
            if agent.option is None:
                random_options[agent_no] = self._random_option(agent)

        return random_options

    def learn(self, reward):
        pass


def create_approximation_model(input_size, num_options):
    inputs = keras.Input(shape=(input_size,))
    option = layers.Dense(num_options, activation="linear")(inputs)

    return keras.Model(inputs=inputs, outputs=option)

