from random import choice

from tensorflow import keras
from tensorflow.keras import layers

OPTIMIZER = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000


action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0



class ZalandoLearner:
    def __init__(self, agents, input_size, num_options):
        self.agents = agents

        self.model_network = create_approximation_model(input_size)
        self.target_network = create_approximation_model(input_size)

        self.action


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

