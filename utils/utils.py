import yaml


def read_yaml(path):
    with open(path, encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config
