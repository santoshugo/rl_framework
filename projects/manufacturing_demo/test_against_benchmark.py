import ray
from ray.rllib.agents import ppo
import numpy as np

from moirai.environment.manufacturing import SingleMachineEnv, JobParams
from rollout import load_ppo_agent, rollout


def minimum_first(environment, param):
    job_queue = environment.job_queue
    schedule_length = environment.schedule_length
    null_action = environment.null_action

    schedule = []
    job_queue = {key: item for key, item in enumerate(job_queue) if item is not None}
    sorted_jobs = sorted(job_queue, key=lambda job_id: getattr(job_queue[job_id], param))

    while len(schedule) < schedule_length:
        try:
            job = sorted_jobs.pop(0)
            schedule.extend([job] * (job_queue[job].processing_time - job_queue[job].done))
        except IndexError:
            schedule.append(null_action)

    return schedule[:schedule_length]


def minimum_slack_first(environment):
    return minimum_first(environment, 'slack')


def earliest_due_date_first(environment):
    return minimum_first(environment, 'due_date')


def earliest_release_date_first(environment):
    return minimum_first(environment, 'due_date')


def shortest_processing_time_first(environment):
    return minimum_first(environment, 'processing_time')


if __name__ == '__main__':
    ray.init()
    ray.tune.register_env('SingleMachineEnv', lambda c: SingleMachineEnv(c))

    agent_config = ppo.DEFAULT_CONFIG.copy()
    agent_config['train_batch_size'] = 500
    agent_config['vf_clip_param'] = 400

    agent_config['env_config'] = {'schedule_length': 14,
                                  'max_job_slots': 100,
                                  'jobs': [JobParams(r_probability=0.15, p={'a': 1, 'b': 3}, d={'a': 3, 'b': 6}),
                                           JobParams(r_probability=0.15, p={'a': 5, 'b': 10}, d={'a': 6, 'b': 11})],
                                  'seed': 99}

    agent = load_ppo_agent(SingleMachineEnv,
                           agent_config,
                           'C:\\Users\\santo\\rl_framework\\projects\\manufacturing_demo\\models\\checkpoint_500\\checkpoint-500')

    benchmarks = {'minimum_slack_first': minimum_slack_first,
                  'earliest_due_date_first': earliest_due_date_first,
                  'earliest_release_date_first': earliest_release_date_first,
                  'shortest_processing_time_first': shortest_processing_time_first}

    env_dict = {'rl': SingleMachineEnv(agent_config['env_config'])}
    obs = env_dict['rl'].reset()
    tasks_completed_dict = {'rl': 0}
    tardiness_dict = {'rl':  list()}

    for benchmark in benchmarks.keys():
        env_dict[benchmark] = SingleMachineEnv(agent_config['env_config'])
        _ = env_dict[benchmark].reset()

        tasks_completed_dict[benchmark] = 0
        tardiness_dict[benchmark] = list()

    for i in range(100):
        action = rollout(agent, obs)
        obs, _, _, info = env_dict['rl'].step(action)

        if info['task_completed']:
            tasks_completed_dict['rl'] += 1
            tardiness_dict['rl'].append(info['tardiness'])

        for key, func in benchmarks.items():
            schedule = func(env_dict[key])
            _, _, _, info = env_dict[key].step(schedule)

            if info['task_completed']:
                tasks_completed_dict[key] += 1
                tardiness_dict[key].append(info['tardiness'])

    for key in tasks_completed_dict.keys():
        print('{} | completed tasks - {}'.format(key, tasks_completed_dict[key]))
        print('{} | tardiness - {}'.format(key, np.mean(tardiness_dict[key])))
        print('\n')

