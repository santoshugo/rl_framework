import pickle
import pprint


def create_config_from_pickle(path):
    with open(path, 'rb') as handle:
        d = pickle.load(handle)

    demand_list = d['demandList']
    machine_list = d['machineList']
    task_list = d['taskList']

    return demand_list, machine_list, task_list

if __name__ == '__main__':
    pp = pprint.PrettyPrinter(width=41, compact=True)
    d, m, t = create_config_from_pickle('C:\\Users\\santo\\rl_framework\\projects\\manufacturing_demo\\files\\demo.pickle')
    print(d)
    print(m)
    print(t)