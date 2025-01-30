import yaml


def load_yaml(filename):
    with open(filename, 'r') as root:
        return yaml.safe_load(root)
