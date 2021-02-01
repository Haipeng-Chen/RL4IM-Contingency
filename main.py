import os
import sys
import copy
import argparse
import collections

import yaml
import numpy as np
import torch as th

from copy import deepcopy
from os.path import dirname, abspath

from src.tasks import REGISTRY as TASK_REGISTRY
from src.utils.logging import get_logger, Logger

from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds


SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("Experiments")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


def _get_basic_config(params, other_params=None, arg_name='graph_name'):
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            if other_params is not None:
                other_params.pop(other_params.index(_v))
            return _v.split("=")[1]
    else:
        raise ValueError('arg_name: {} should be set in command arguments, '
                         'it is not found in {}'.format(arg_name, params))


def _get_config(params, arg_name, subfolder=None):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "src", "tasks", "config", "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["use_cuda"] and th.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config["gpu_id"])

    return config


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['seed'] = config["seed"]

    # save pid
    config['pid'] = os.getpid()

    # if config["use_cuda"]:
    #     th.backends.cudnn.deterministic = True
    #     th.backends.cudnn.benchmark = False

    args_sanity_check(config, _log)

    logger = Logger(_log)

    config['local_results_path'] = os.path.join(config['local_results_path'], _run._id)

    if config['use_tensorboard']:
        logger.setup_tb(config['local_results_path'])

    # sacred is on by default
    logger.setup_sacred(_run)

    # run the task
    task_runner = TASK_REGISTRY[config['task']]
    task_runner(_run, config, logger, run_args=None)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    params_bak = deepcopy(sys.argv)

    # Get the defaults from task_default.yaml
    with open(os.path.join(os.path.dirname(__file__), "src", "tasks", "config", "task_default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "task_default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config")
    alg_config = _get_config(params, "--config")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    results_path = _get_basic_config(params_bak, other_params=params, arg_name='--results-dir')
    task_name = _get_basic_config(params_bak, arg_name='--config')
    # Save to disk by default for sacred
    save_path = os.path.join(results_path, task_name)

    logger.info(f"Saving to FileStorageObserver in {save_path}/sacred.")
    file_obs_path = os.path.join(save_path, "sacred")

    # save the results to config_dict and can be used in args
    config_dict['local_results_path'] = file_obs_path

    # now add all the config to sacred
    ex.add_config(config_dict)
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)  # call my_main(_run, _config, _log)
