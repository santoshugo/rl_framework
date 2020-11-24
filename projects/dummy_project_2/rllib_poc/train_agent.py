import argparse
import os

from gym.spaces import Discrete, MultiDiscrete, Dict, Box, Tuple
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.rllib.agents import dqn, pg, ppo, a3c
import ray

from projects.dummy_project_2.rllib_poc.env import ZalandoEnvironment, ParametricActionsModel


if __name__ == '__main__':
    ray.init()
    register_env(
        "zalando_env",
        lambda config: ZalandoEnvironment(config)
    )

    # obs_space = Tuple((Discrete(9 + 1),  # nodes + not in node
    #                    Discrete(12 + 1),  # edges + not in edge
    #                    Discrete(2),  # carrying full
    #                    Discrete(2),  # carrying empty
    #                    Box(low=-1, high=2, shape=(1,))  # battery
    #                    ))

    obs_space = Dict({
        "action_mask": Box(low=0, high=1, shape=(13,)),
        "avail_actions": Box(low=0, high=1, shape=(13,)),
        "real_obs": Box(low=0, high=1, shape=(9 + 1 + 12 + 1 + 2 + 2 + 1,))
    })
    act_space = Discrete(9 + 4)

    ModelCatalog.register_custom_model(
        "model", ParametricActionsModel)

    config = pg.DEFAULT_CONFIG

    config["num_workers"] = 8
    config["seed"] = 14
    config["multiagent"]["policies"] = {"pol{}".format(i): (None, obs_space, act_space, {"gamma": 0.99, "agent_id": i}) for i in range(2)}
    config["multiagent"]["policy_mapping_fn"] = lambda agent_id: 'pol{}'.format(agent_id)

    config["model"]["custom_model"] = "model"

    # config["lr"] = 0.01
    config["horizon"] = 1000
    config['no_done_at_end'] = True
    config["env_config"] = {"num_agents": 2,
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
                            }

    # config["exploration_config"]["epsilon_timesteps"] = 200000

    trainer = pg.PGTrainer(env="zalando_env", config=config)

    for i in range(1, 1000):
        results = trainer.train()

        if i % 10 == 0:  # save every 10th training iteration
            print(i, results)
            #checkpoint_path = trainer.save()
            #print(checkpoint_path)

    ray.shutdown()
