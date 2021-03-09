import ray
from ray.rllib.agents import ppo
from rollout import load_ppo_agent, rollout
from moirai.environment import ManufacturingDispatchingEnv
from gym.spaces import Dict

def least_slack_time(job_queue):
    pass


if __name__ == '__main__':
    pass

if __name__ == '__main__':
    ray.init()
    ray.tune.register_env('ManufacturingDispatchingEnv', lambda config: ManufacturingDispatchingEnv(config))

    agent = load_ppo_agent(ManufacturingDispatchingEnv,
                           ppo.DEFAULT_CONFIG.copy(),
                           'C:\\Users\\santo\\rl_framework\\projects\\manufacturing_demo\\models\\checkpoint_10\\checkpoint-10')

    env = ManufacturingDispatchingEnv({})
    obs = env.reset()
    print(obs)
    action = rollout(agent, obs)
    print(action)

    action = rollout(agent, obs)
    print(action)

    action = rollout(agent, obs)
    print(action)

    action = rollout(agent, obs)
    print(action)

    action = rollout(agent, obs)
    print(action)


