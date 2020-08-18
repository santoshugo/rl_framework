import copy
import json

if __name__ == '__main__':
    json_file = dict()

    json_file["no_agents"] = 10
    json_file["width"] = 204
    json_file["height"] = 6

    environment = list()
    for x in range(0, 205):
        d = dict()
        d['x'] = x
        d['y'] = 1
        d['pickup_full_1'] = None
        d['pickup_full_2'] = None
        d['pickup_full_3'] = None
        d['pickup_empty_1'] = None
        d['pickup_empty_2'] = None
        d['drop_empty'] = None
        d['drop_full'] = None
        d['charging_station'] = None

        environment.append(d)
        d2 = copy.deepcopy(d)
        d2['y'] = 6
        environment.append(d2)

    json_file['environment'] = environment
    with open('../../docs/maps/dummy_map_2.json', 'w') as fp:
        json.dump(json_file, fp)
