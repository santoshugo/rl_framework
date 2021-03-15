import numpy as np
from gym.spaces import Box, Dict, MultiDiscrete
from ray.rllib.utils.spaces.repeated import Repeated


class SingleMachineObs:
    def __init__(self, environment):
        self.environment = environment
        self.machine_state = MultiDiscrete([2] * self.environment.schedule_length)
        self.job_state = Repeated(Box(low=-self.environment.max_steps_per_iterations,
                                      high=self.environment.max_steps_per_iterations,
                                      shape=(3,),
                                      dtype=np.int),
                                  max_len=self.environment.max_job_slots)

        self.observation_space = Dict({'machine_state': self.machine_state,
                                       'job_state': self.job_state})

    def get_observation(self, job_queue, action):
        machine_state = [1 if a != self.environment.null_action else 0 for a in action]
        job_state = []

        for job in job_queue:
            processing_time = job.processing_time
            slack = job.slack
            done = job.done

            job_state.append([processing_time, slack, done])

        return {'machine_state': machine_state,
                'job_state': job_state}


        # processing_time = [self.environment.job_queue[i].processing_time if self.environment.job_queue[i] is not None else 0 for i in range(self.environment.max_job_slots)]
        # slack_state = list()
        #
        # for i in range(self.environment.max_job_slots):
        #     if self.environment.job_queue[i] is None:
        #         slack_state.append(self.environment.max_steps_per_iterations)
        #     elif i not in action:
        #         completion_date = self.environment.time_step + len(action)
        #         slack_state.append(completion_date - self.environment.job_queue[i].due_date)
        #     else:
        #         completion_date = self.environment.time_step + len(action) - action[::-1].index(i)
        #         slack_state.append(completion_date - self.environment.job_queue[i].due_date)
        #
        # return np.array(machine_state + processing_time + slack_state)