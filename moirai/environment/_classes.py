from typing import List
from collections import namedtuple

from gym import Env
from gym.spaces import MultiDiscrete, Box
import numpy as np

Job = namedtuple('Task', ['length', 'slack', 'done', 'id'])

# slack array must be computed
FULL_BACKLOG_PENALTY = -10


class ManufacturingDispatchingEnv(Env):
    def __init__(self, config):
        self.time_steps = config.get('time_steps', 15)
        self.slack_array = config.get('slack_array', 5)
        self.n_job_slots = config.get('n_job_slots', 10)
        self.n_backlog_slots = config.get('n_backlog_slots', 60)
        self.seed = config.get('seed', 2)
        self.metric = config.get('metric', 'tardiness')

        assert self.metric in ['tardiness', 'lateness'], 'Acceptable metrics are lateness and tardiness'
        self.max_steps_per_iterations = config.get('max_steps_per_iterations', 100)
        self.rng = np.random.default_rng(self.seed)

        self.action_space = MultiDiscrete([self.n_job_slots + 1] * self.time_steps)
        self.observation_space = Box(-200, 200, shape=(self.time_steps + self.n_job_slots + 1 + self.n_job_slots,), dtype=np.int)  # hardcoded lower/upper bounds for now
        self.reward_function = self.tardiness if self.metric == 'tardiness' else self.lateness

        self.null_action = self.n_job_slots

        self.current_job = None

        self.backlog = list()
        self.job_queue = {i: None for i in range(self.n_job_slots)}
        self.job_queue_empty_slots = set(list(range(self.n_job_slots)))

        self.i = 0
        self.job_id = 0

    def reset(self):
        self.current_job = self.null_action

        self.backlog = list()
        self.job_queue = {i: None for i in range(self.n_job_slots)}
        self.job_queue_empty_slots = set(list(range(self.n_job_slots)))

        self.i = 0
        self.job_id = 0

        return self.get_observation([self.null_action] * self.time_steps)

    def step(self, action: list):
        r = 0  # add rewards later
        action = self.schedule_correction(action)  # assume that it is valid
        info = {'task_completed': False, 'lateness': None, 'tardiness': None}
        # if doing nothing assign first task in schedule
        if self.current_job == self.null_action:
            self.current_job = action[0]

        # update task slack
        for job in self.job_queue:
            if self.job_queue[job] is not None:
                slack = self.job_queue[job].slack
                self.job_queue[job] = self.job_queue[job]._replace(slack=slack - 1)

        for job in range(len(self.backlog)):
            slack = self.backlog[job].slack
            self.backlog[job] = self.backlog[job]._replace(slack=slack - 1)

        # update current task
        if self.current_job != self.null_action:
            done = self.job_queue[self.current_job].done
            self.job_queue[self.current_job] = self.job_queue[self.current_job]._replace(done=done + 1)

            # update finished tasks
            if self.job_queue[self.current_job].done == self.job_queue[self.current_job].length:
                info = {'task_completed': True, 'lateness': abs(self.job_queue[self.current_job].slack), 'tardiness': max(-self.job_queue[self.current_job].slack, 0)}

                # remove job from queue
                self.job_queue[self.current_job] = None
                self.job_queue_empty_slots.add(self.current_job)
                # move task from backlog to job queue
                if self.backlog:
                    self.job_queue[min(self.job_queue_empty_slots)] = self.backlog.pop(0)
                    self.job_queue_empty_slots.remove(min(self.job_queue_empty_slots))
                # change current task in execution to null job
                self.current_job = self.null_action

        # set reward
        for i, job in self.job_queue.items():
            if job is None:
                continue
            elif i not in action and job.slack > self.time_steps + job.length:
                pass  # is delivery date is too distant then it is not supposed to be in scheduling
            elif i not in action and job.slack <= self.time_steps:
                r -= self.reward_function(self.time_steps + job.length, job.slack)  # assumes task will be started just after current scheduling
            elif i in action:
                completion_date = len(action) - action[::-1].index(i)  # gets date when job is completed
                r -= self.reward_function(completion_date, job.slack)
            else:
                raise Exception('Unexpected case')

        # create task
        job = self.create_job()

        if job is not None:
            if len(self.job_queue_empty_slots) > 0:
                pos = min(self.job_queue_empty_slots)
                self.job_queue[pos] = job
                self.job_queue_empty_slots.remove(pos)
            elif len(self.backlog) < self.n_backlog_slots:
                self.backlog.append(job)
            else:
                r += FULL_BACKLOG_PENALTY

        # create observation
        observation = self.get_observation(action)

        # check if done
        self.i += 1
        done = True if self.i >= self.max_steps_per_iterations else False
        return observation, r, done, info

    def get_observation(self, action):
        machine_state = [1 if a != self.null_action else 0 for a in action]
        processing_time =[self.job_queue[i].length if self.job_queue[i] is not None else 0 for i in range(self.n_job_slots)]
        backlog_state = [len(self.backlog)]
        slack_state = list()

        for i in range(self.n_job_slots):
            if self.job_queue[i] is None:
                slack_state.append(0)
            elif i not in action and self.job_queue[i].slack > self.time_steps + self.job_queue[i].length:
                slack_state.append(0)
            elif i not in action and self.job_queue[i].slack <= self.time_steps + self.job_queue[i].length:
                slack_state.append(self.job_queue[i].slack - self.time_steps + self.job_queue[i].length)
            else:
                completion_date = len(action) - action[::-1].index(i)
                slack_state.append(self.job_queue[i].slack - completion_date)

        return np.array(machine_state + processing_time + backlog_state + slack_state)

    def lateness(self, completion_date: int, due_date: int) -> int:
        return abs(completion_date - due_date)

    def tardiness(self, completion_date: int, due_date: int) -> int:
        return max(completion_date - due_date, 0)

    def schedule_correction(self, action: List) -> List:
        corrected_action = []

        if self.current_job != self.null_action:
            corrected_action += [self.current_job] * (self.job_queue[self.current_job].length - self.job_queue[self.current_job].length)

        i = len(corrected_action)
        while i < self.time_steps:
            j = action[i]
            if j == self.null_action:  # case where action is none
                corrected_action.append(self.null_action)
                i += 1
            elif j in self.job_queue_empty_slots:  # case where action is invalid
                corrected_action.append(self.null_action)
                i += 1
            elif j not in self.job_queue_empty_slots and j not in corrected_action:
                time_steps = min(self.job_queue[j].length, self.time_steps - i)
                corrected_action += [j] * time_steps
                i += time_steps
            else:
                corrected_action.append(self.null_action)
                i += 1

        return corrected_action

    def create_job(self, arrival_speed=0.5, p_small=0.8, p_urgent=0.5):
        if self.rng.random() > arrival_speed:
            return None

        if self.rng.random() <= p_small:
            length = self.rng.integers(1, 3)
        else:
            length = self.rng.integers(6, 11)

        if self.rng.random() <= p_urgent:
            slack = self.rng.integers(1, 6)
        else:
            slack = self.rng.integers(5, 11)

        self.job_id += 1
        return Job(length=length, slack=slack, done=0, id=self.job_id)
