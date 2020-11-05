import argparse
from gym.spaces import Tuple, MultiDiscrete, Dict, Discrete
import os

import ray
from ray import tune
from ray.tune import register_env, grid_search
from ray.rllib.env.multi_agent_env import ENV_STATE
from ray.rllib.examples.env.two_step_game import TwoStepGame
from ray.rllib.utils.test_utils import check_learning_achieved

from projects.dummy_project_2.rllib_poc.simple_env import SimpleEnvironment


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PG")
parser.add_argument("--num-cpus", type=int, default=0)
# parser.add_argument("--as-test", action="store_true")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--stop-reward", type=float, default=7.0)
parser.add_argument("--stop-timesteps", type=int, default=500)


if __name__ == '__main__':
    args = parser.parse_args()

    register_env(
        "simple_env",
        lambda config: SimpleEnvironment(config)
    )

    config = {
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": "tf",
    }
    group = False

    ray.init(_node_ip_address="127.0.0.1")

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
    }

    config = dict(config, **{
        "env": "simple_env",
    })

    results = tune.run(args.run, stop=stop, config=config, verbose=1)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

    ray.shutdown()