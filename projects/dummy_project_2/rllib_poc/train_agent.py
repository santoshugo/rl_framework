import argparse
import os

from gym.spaces import Discrete, MultiDiscrete, Dict, Box, Tuple
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.rllib.agents import dqn, pg, ppo, a3c
import ray

from projects.dummy_project_2.rllib_poc.env import ZalandoEnvironment

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PG")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-reward", type=float, default=7.0)
parser.add_argument("--stop-timesteps", type=int, default=50000)

if __name__ == '__main__':
    # ray.init()
    register_env(
        "zalando_env",
        lambda config: ZalandoEnvironment(config)
    )

    obs_space = Tuple((Discrete(9 + 1),  # nodes + not in node
                               Discrete(12 + 1),  # edges + not in edge
                               Discrete(2),  # carrying full
                               Discrete(2),  # carrying empty
                               Box(low=-1, high=2, shape=(1,))  # battery
                               ))

    act_space = Discrete(9 + 4)

    config = {
        "multiagent": {
            "policies": {"pol{}".format(i): (None, obs_space, act_space, {"gamma": 0.99, "agent_id": i}) for i in range(2)},
            "policy_mapping_fn": lambda agent_id: 'pol{}'.format(agent_id)
        },
        "lr": 0.01,
        "horizon": 100,
        "no_done_at_end": True,
        "env_config": {"num_agents": 2,
                       "agent_speed": 0.5,
                       "initial_node": {0: 1,
                                        1: 1,
                                        # 2: (1, 'node'),
                                        # 3: (1, 'node'),
                                        # 4: (1, 'node'),
                                        # 5: (5, 'node'),
                                        # 6: (5, 'node'),
                                        # 7: (5, 'node'),
                                        # 8: (5, 'node'),
                                        # 9: (5, 'node')
                                        }
                       }}
    print(config)

    trainer = dqn.DQNTrainer(env="zalando_env", config=config)

    for i in range(1000):
        results = trainer.train()

        if i % 10 == 0:  # save every 10th training iteration
            print(results)
            checkpoint_path = trainer.save()
            print(checkpoint_path)

    # ray.shutdown()
