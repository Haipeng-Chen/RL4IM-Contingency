import copy
import argparse

from src.tasks import REGISTRY as TASK_REGISTRY


def parser():
    parser = argparse.ArgumentParser(description='Arguments of influence maximzation')
    parser.add_argument('--task', type=str, default='colge', choices=['basic_qdn', 'colge'])
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--results-dir', type=str, default='./results')

    return parser


if __name__ == '__main__':
    parser = parser()
    #task_runner = TASK_REGISTRY[copy.deepcopy(parser).parse_args().task]
    task_runner = TASK_REGISTRY[parser.parse_args().task]
    task_runner(parser, run_args=None)
