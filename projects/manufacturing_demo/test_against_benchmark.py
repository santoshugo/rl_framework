import ray
from ray.rllib.agents import ppo
from rollout import load_ppo_agent, rollout
from moirai.environment import ManufacturingDispatchingEnv
import numpy as np


def least_slack_first(job_queue, time_steps, null_action):
    schedule = []
    job_queue = {key: item for key, item in job_queue.items() if item is not None}
    sorted_jobs = sorted(job_queue, key=lambda job_id: job_queue[job_id].slack)

    while len(schedule) < time_steps:
        try:
            job = sorted_jobs.pop(0)
            schedule.extend([job] * (job_queue[job].length - job_queue[job].done))
        except IndexError:
            schedule.append(null_action)

    return schedule[:time_steps]


if __name__ == '__main__':
    ray.init()
    ray.tune.register_env('ManufacturingDispatchingEnv', lambda config: ManufacturingDispatchingEnv(config))

    agent = load_ppo_agent(ManufacturingDispatchingEnv,
                           ppo.DEFAULT_CONFIG.copy(),
                           'C:\\Users\\santo\\rl_framework\\projects\\manufacturing_demo\\models\\checkpoint_500\\checkpoint-500')

    env_lsf = ManufacturingDispatchingEnv({})
    env_rl = ManufacturingDispatchingEnv({})

    _ = env_lsf.reset()
    obs = env_rl.reset()

    tasks_completed_lsf = 0
    tasks_completed_rl = 0
    tardiness_lsf = []
    tardiness_rl = []

    for i in range(100):
        schedule_lsf = least_slack_first(env_lsf.job_queue, env_lsf.time_steps, env_lsf.null_action)
        action = rollout(agent, obs)

        _, _, _, info_lsf = env_lsf.step(schedule_lsf)
        obs, _, _, info_rl = env_rl.step(action)

        if info_lsf['task_completed']:
            tasks_completed_lsf += 1
            tardiness_lsf.append(info_lsf['tardiness'])

        if info_rl['task_completed']:
            tasks_completed_rl += 1
            tardiness_rl.append(info_rl['tardiness'])

    print('least slack first | completed tasks - ', tasks_completed_lsf)
    print('least slack first | mean tardiness - ', np.mean(tardiness_lsf))

    print('RL | completed tasks - ', tasks_completed_rl)
    print('RL | mean tardiness - ', np.mean(tardiness_rl))

