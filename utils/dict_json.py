import json

def filter_dict(dict_sample, keys):
    return dict([(i, dict_sample[i]) for i in dict_sample if i in keys])

def write_dict_to_json(dict_sample, file_path):
    with open(file_path, 'w') as file:
        json_string = json.dumps(dict_sample, indent=4)
        file.write(json_string)

def read_json_to_dict(file_path):
    with open(file_path) as file:
        dict_sample = json.load(file)
    return dict_sample