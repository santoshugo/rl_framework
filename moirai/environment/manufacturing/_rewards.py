def single_machine_reward(environment, action):
    r = 0
    for i, job in enumerate(environment.job_queue):
        if job is None:
            continue
        elif i not in action:  # if job not in schedule assumes it will start just after current schedule is finished
            r -= max(environment.schedule_length + job.processing_time - job.due_date, 0)
        elif i in action:  # gets date when job is completed
            completion_date = len(action) - action[::-1].index(i)
            r -= max(completion_date - job.due_date - environment.time_step, 0)
        else:  # just in case something unexpected happens
            raise Exception('Unexpected case')

    return r
