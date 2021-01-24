from src.tasks.task_basic_dqn import basic_dqn
from src.tasks.task_colge import run_colge


REGISTRY = {}

REGISTRY["basic_qdn"] = basic_dqn
REGISTRY["colge"] = run_colge
