from os.path import dirname, realpath, join
import time

import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print

from moirai.environment import ManufacturingDispatchingEnv

DATA_FOLDER = join(dirname(realpath(__file__)), 'models')
NUM_EPOCHS = 500


if __name__ == '__main__':

    ray.init()
    ray.tune.register_env('ManufacturingDispatchingEnv', lambda config: ManufacturingDispatchingEnv(config))

    config = ppo.DEFAULT_CONFIG.copy()
    config['train_batch_size'] = 500
    config['vf_clip_param'] = 400
    trainer = ppo.PPOTrainer(config=config, env='ManufacturingDispatchingEnv')

    start = time.time()
    for i in range(NUM_EPOCHS):
        # Perform one iteration of training the policy with PPO
        result = trainer.train()

        if i % 20 == 0:
            print('\n--- {} ---'.format(i))
            print(pretty_print(result))

    end = time.time()
    print('time: ', end - start)

    checkpoint = trainer.save(DATA_FOLDER)
