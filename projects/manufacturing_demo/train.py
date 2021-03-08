from os.path import dirname, realpath, join

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

from moirai.environment import ManufacturingDispatchingEnv

DATA_FOLDER = join(dirname(realpath(__file__)), 'models')
NUM_EPOCHS = 100

if __name__ == '__main__':
    ray.init()
    ray.tune.register_env('ManufacturingDispatchingEnv', lambda config: ManufacturingDispatchingEnv(config))

    config = ppo.DEFAULT_CONFIG.copy()
    trainer = ppo.PPOTrainer(config=config, env='ManufacturingDispatchingEnv')

    for i in range(NUM_EPOCHS):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()

    checkpoint = trainer.save(DATA_FOLDER)
