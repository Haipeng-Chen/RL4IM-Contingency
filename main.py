import copy
import argparse

from src.tasks import REGISTRY as TASK_REGISTRY


def parser():
    parser = argparse.ArgumentParser(description='Arguments of influence maximzation')
    parser.add_argument('--task', type=str, default='colge', choices=['basic_qdn', 'colge'])

    return parser


if __name__ == '__main__':
    parser = parser()
    task_runner = TASK_REGISTRY[copy.deepcopy(parser).parse_args().task]
    task_runner(parser, run_args=None)
