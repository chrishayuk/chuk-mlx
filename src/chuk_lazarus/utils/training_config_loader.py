import yaml


def load_training_config(config_file):
    # load the training config from yaml
    with open(config_file) as file:
        # return the yaml
        return yaml.safe_load(file)
