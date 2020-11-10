import argparse
import os

from gym.spaces import Discrete, MultiDiscrete, Dict, Box
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from ray.rllib.agents import dqn, pg, ppo, a3c

from projects.dummy_project_2.rllib_poc.simple_env import SimpleEnvironment, ParametricActionsModel


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PG")
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--stop-reward", type=float, default=7.0)
parser.add_argument("--stop-timesteps", type=int, default=50000)


if __name__ == '__main__':

    register_env(
        "simple_env",
        lambda config: SimpleEnvironment(config)
    )

    ModelCatalog.register_custom_model(
        "model", ParametricActionsModel)

    obs_space = Dict({
            "action_mask": Box(0, 1, shape=(7,)),
            "avail_actions": Box(0, 1, shape=(7,)),
            "real_obs": MultiDiscrete([5, 2])
        })

    act_space = Discrete(7)

    trainer = pg.PGTrainer(env="simple_env", config={
        "model": {
            "custom_model": "model",
        },
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.

        },
        "multiagent": {
            "policies": {
                # the first tuple value is None -> uses default policy
                'pol1': (None, obs_space, act_space, {"gamma": 1}),
                'pol2': (None, obs_space, act_space, {"gamma": 1})
            },
            "policy_mapping_fn": lambda agent_id: 'pol1' if agent_id == 1 else 'pol2'
        },
    })

    for i in range(1000):
        print(i, trainer.train())
