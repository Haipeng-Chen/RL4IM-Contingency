import sys
import argparse

import numpy as np
import networkx as nx

from types import SimpleNamespace

from src.runner import runners
from src.agent.colge.agent import Agent
from src.environment.graph import Graph
from src.environment.env import NetworkEnv, Environment
from src.environment.colge import env as colge_env
from src.tasks.task_basic_dqn import custom_env_parser, get_graph


def run_colge(_run, config, _log, run_args=None):
    args = SimpleNamespace(**config)
    graph_dic = {}
    #seed = 125
    #graph_one = graph.Graph(graph_type=args.graph_type, cur_n=20, p=0.15,m=4, seed=seed)
    
    for graph_ in range(args.graph_nbr):
        seed = np.random.seed(120+graph_)
        G, g, graph_name = get_graph(args.graph_index)
        graph_dic[graph_] = Graph.create_graph(g)

    agent_class = Agent(graph_dic, args.model, args.lr, args.bs, args.n_step)

    # env_class = colge_env.Environment(graph_dic, args.environment_name)
    env_class = Environment(G=g, cascade=args.cascade, T=args.T, budget=args.budget, 
                           propagate_p=args.propagate_p, l=args.l, d=args.d, q=args.q)

    my_runner = runners.Runner(args, env_class, agent_class, args.verbose)
    final_reward = my_runner.loop(args.ngames, args.epoch, args.niter)
