#from src.tasks.task_basic_dqn import basic_dqn
from src.tasks.task_rl4im import run_rl4im


REGISTRY = {}

#REGISTRY["basic_dqn"] = basic_dqn
REGISTRY["rl4im"] = run_rl4im
