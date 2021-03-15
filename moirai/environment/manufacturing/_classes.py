from typing import List
from collections import namedtuple

from gym import Env
from gym.spaces import MultiDiscrete, Box
from ray.rllib.utils.spaces.repeated import Repeated
import numpy as np

from ._utils import JobCreator
from ._observations import SingleMachineObs
from ._rewards import single_machine_reward

Job = namedtuple('Task', ['length', 'slack', 'done', 'id'])


FULL_JOB_SLOT_PENALTY = -10


class Machine:
    def __init__(self, machine_id, null_action):
        self.machine_id = machine_id
        self.null_action = null_action

    def step(self, time_step, job_idx, job_queue):
        if job_idx >= len(job_queue) and job_idx != self.null_action:
            raise Exception('Job to run in machine {} not in job queue'.format(self.machine_id))

        info = {'task_completed': False, 'tardiness': None}

        if job_idx != self.null_action:
            done = job_queue[job_idx].done + 1
            job_queue[job_idx] = job_queue[job_idx]._replace(done=done)

            if done == job_queue[job_idx].processing_time:
                info = {'task_completed': True,
                        'tardiness': max(time_step - job_queue[job_idx].due_date, 0)}

                job_queue.pop(job_idx)

        return job_queue, info


class SingleMachineEnv(Env):
    def __init__(self, config):
        self.schedule_length = config.get('schedule_length', 15)
        self.max_job_slots = config.get('max_job_slots', 50)
        self.max_steps_per_iterations = config.get('max_steps_per_iterations', 100)
        self.job_params = config.get('jobs')
        self.seed = config.get('seed', None)

        self.reward_function = single_machine_reward

        self.observation = SingleMachineObs(self)
        self.observation_function = self.observation.get_observation

        self.action_space = MultiDiscrete([self.max_job_slots + 1] * self.schedule_length)
        self.observation_space = self.observation.observation_space

        self.null_action = self.max_job_slots
        self.current_job = self.null_action

        self.job_queue = []

        self.job_creator = JobCreator(self.job_params, seed=self.seed)
        self.time_step = 0

        self.machine = Machine(machine_id=0, null_action=self.null_action)

    def reset(self):
        self.time_step = 0
        self.current_job = self.null_action

        self.job_queue = []

        self.job_creator.reset()

        return self.observation_function(self.job_queue, [self.null_action] * self.schedule_length)

    def step(self, action: list):
        action = self.schedule_correction(action)

        # if doing nothing assign first task in schedule
        if self.current_job == self.null_action:
            self.current_job = action[0]

        # update task slack
        for i, job in enumerate(self.job_queue):
            slack = job.slack
            self.job_queue[i] = job._replace(slack=slack - 1)

        self.job_queue, info = self.machine.step(self.time_step, self.current_job, self.job_queue)
        if info['task_completed']:
            self.current_job = self.null_action

        r = self.reward_function(self, action)

        # create task
        for job in self.job_creator.create():
            if len(self.job_queue) < self.max_job_slots:
                self.job_queue.append(job)
            else:
                r += FULL_JOB_SLOT_PENALTY

        # create observation
        observation = self.observation_function(self.job_queue, action)

        # check if done
        self.time_step += 1
        done = True if self.time_step >= self.max_steps_per_iterations else False

        return observation, r, done, info

    def schedule_correction(self, action: List) -> List:
        """
        Corrects the schedule (action) received to a valid one that can be used by the environment
        :param action: input schedule
        :return: valid schedule
        """
        corrected_action = []

        if self.current_job != self.null_action:  # If a job is already started it must be finished before another one is started
            corrected_action += [self.current_job] * (self.job_queue[self.current_job].processing_time - self.job_queue[self.current_job].done)

        i = len(corrected_action)
        while i < self.schedule_length:
            j = action[i]
            if j == self.null_action:  # case where action is none
                corrected_action.append(self.null_action)
                i += 1
            elif j >= len(self.job_queue):  # case where action is invalid
                corrected_action.append(self.null_action)
                i += 1
            elif j < len(self.job_queue) and j not in corrected_action:  # case where action is valid, so a new job is added with the correct length
                time_steps = min(self.job_queue[j].processing_time, self.schedule_length - i)
                corrected_action += [j] * time_steps
                i += time_steps
            else:
                corrected_action.append(self.null_action)
                i += 1

        return corrected_action
