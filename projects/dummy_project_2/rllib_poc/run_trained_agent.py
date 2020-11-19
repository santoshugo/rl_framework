from ray.rllib.agents import dqn
from gym.spaces import Tuple, Discrete, Box
from ray.tune import register_env
import ray
import numpy as np

from projects.dummy_project_2.rllib_poc.env import ZalandoEnvironment

PATH = '/home/hugo/ray_results/DQN_zalando_env_2020-11-18_18-06-03vhi9d2fr/checkpoint_500/checkpoint-500'

if __name__ == '__main__':
    ray.init()
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

    trainer = dqn.DQNTrainer(env="zalando_env", config=config)
    trainer.restore(PATH)

    test_observation = [0, 12, False, False, [1]]

    action = trainer.compute_action(test_observation, policy_id='pol0', explore=False)
    print(action)

    ray.shutdown()
