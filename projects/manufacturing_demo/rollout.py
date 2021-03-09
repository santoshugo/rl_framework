import ray.rllib.agents.ppo as ppo


def load_ppo_agent(env_class, config, path):

    agent = ppo.PPOTrainer(config=config, env=env_class)
    agent.restore(path)

    return agent


def rollout(agent, obs):
    return agent.compute_action(obs, explore=False)


