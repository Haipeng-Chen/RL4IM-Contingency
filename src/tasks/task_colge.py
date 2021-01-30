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
from src.tasks.task_basic_dqn import get_graph


def run_colge(_run, config, logger, run_args=None):
    args = SimpleNamespace(**config)
    graph_dic = {}
    #seed = 125
    #graph_one = graph.Graph(graph_type=args.graph_type, cur_n=20, p=0.15,m=4, seed=seed)
    
    for graph_ in range(args.graph_nbr):
        G, g, graph_name = get_graph(args.graph_index)
        graph_dic[graph_] = Graph.create_graph(g)
        graph_dic[graph_].graph_name = graph_name

    agent_class = Agent(graph_dic, args.model, args.lr, args.bs, args.n_step, args=args)
    if args.use_cuda:
        agent_class.cuda()
    # env_class = colge_env.Environment(graph_dic, args.environment_name)
    env_class = Environment(cascade=args.cascade, T=args.T, budget=args.budget, 
                           propagate_p=args.propagate_p, l=args.l, d=args.d, q=args.q, graphs=graph_dic)

    my_runner = runners.Runner(args, env_class, agent_class, args.verbose, logger=logger)
    final_reward = my_runner.loop()
