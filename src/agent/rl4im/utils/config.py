import os
import yaml

from pathlib import Path


def load_model_config():
    path = Path(os.path.realpath(__file__)).parent.parent
    with open(os.path.join(path, "model_config.yml")) as config_file:
        model_config = yaml.load(config_file)
        # Todo: check_model_config(model_config)

    return model_config
