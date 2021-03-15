import numpy as np
from typing import List
from collections import namedtuple

Job = namedtuple('Job', ['processing_time', 'release_date', 'due_date', 'slack', 'done', 'id'])
JobParams = namedtuple('JobParams', ['r_probability', 'p', 'd'])


class JobCreator:
    def __init__(self, job_params: List[JobParams], seed: int):
        self.seed = seed

        self.job_id = 0
        self.time_step = 0
        self.rng = np.random.default_rng(self.seed)

        self.job_params = job_params

    def reset(self):
        self.job_id = 0
        self.time_step = 0
        self.rng = np.random.default_rng(self.seed)

    def create(self) -> List:
        job_list = []
        for param in self.job_params:
            if self.rng.random() > param.r_probability:
                continue
            else:
                processing_time = self.rng.integers(param.p['a'], param.p['b'])
                due_date = self.time_step + self.rng.integers(param.d['a'], param.d['b'])

                job_list.append(Job(processing_time=processing_time,
                                    release_date=self.time_step,
                                    due_date=due_date,
                                    slack=due_date - processing_time - self.time_step,
                                    done=0,
                                    id=self.job_id))
                self.job_id += 1

        self.time_step += 1

        return job_list




