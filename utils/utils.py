import os

import yaml
import json


def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config


def save_config(config, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_dir + 'config.json', "w") as f:
        json.dump(config, f)
    return
