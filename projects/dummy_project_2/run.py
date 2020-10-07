import json

from projects.dummy_project_2.agent import ZalandoAgent
from projects.dummy_project_2.environment import ZalandoEnvironment, ZalandoObservation


if __name__ == '__main__':
    dummy_map_file = 'C:\\Users\\santosh\\PycharmProjects\\rl_framework\\docs\\maps\\dummy_map_2.json'

    with open(dummy_map_file) as f:
        env_map = json.load(f)

    # refills each cart with probability 1 / n
    pickup_refill = {0: 20, 3: 20, 6: 20, 7: 20, 8: 20}

    initial_state = {0: (1, 'node'),
                     1: (1, 'node'),
                     2: (1, 'node'),
                     3: (1, 'node'),
                     4: (1, 'node'),
                     5: (5, 'node'),
                     6: (5, 'node'),
                     7: (5, 'node'),
                     8: (5, 'node'),
                     9: (5, 'node')
                     }


    def battery_decay_function(x):
        return x - 100 / (90 * 60)

    def battery_charge_function(x):
        return x + 100 / (180 * 60)

    agents = {agent_no: ZalandoAgent(agent_no, 0.5, battery_decay_function, battery_charge_function) for agent_no in range(10)}

    env = ZalandoEnvironment(env_map, agents, ZalandoObservation, initial_nodes, pickup_refill)
    obs = env.reset()

    actions = {0: 0, 1: 2, 3: -1, 4: -1, 5: 4, 6: -1, 7: -1, 8: -1, 9: -1}
    obs, reward = env.step(actions)

    print(obs)
    print(reward)
    print('\n')

    actions = {0: -2, 1: -2, 3: -1, 4: -1, 5: 4, 6: -1, 7: -1, 8: -1, 9: -1}
    for k in range(100):
        obs, reward = env.step(actions)

        print(obs)
        print(reward)
        print('\n')