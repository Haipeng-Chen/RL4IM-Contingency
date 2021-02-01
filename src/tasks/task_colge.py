import sys
import argparse

import numpy as np
import networkx as nx

from types import SimpleNamespace

from src.runner import runners
from src.agent.colge.agent import DQAgent
from src.agent.baseline import *
from src.environment.graph import Graph
from src.environment.env import NetworkEnv, Environment
from src.tasks.task_basic_dqn import get_graph
import ipdb


def run_colge(_run, config, logger, run_args=None):
    args = SimpleNamespace(**config)
    print('Loading graph: ', args.graph_type)
    graph_dic = {}
    #seed = 125
    #graph_one = graph.Graph(graph_type=args.graph_type, cur_n=20, p=0.15,m=4, seed=seed)
    
    for graph_ in range(args.graph_nbr):
        #G, g, graph_name = get_graph(args.graph_index)
        #graph_dic[graph_] = Graph.create_graph(g)
        #graph_dic[graph_].graph_name = graph_name
        seed = graph_ #fix seed for now; TODO: revise later
        graph_dic[graph_]=Graph(graph_type=args.graph_type, cur_n=args.node, p=args.p,m=args.m,seed=seed)
        graph_dic[graph_].graph_name = str(graph_)

    env_class = Environment(cascade=args.cascade, T=args.T, budget=args.budget,
                           propagate_p=args.propagate_p, l=args.l, d=args.d, q=args.q, graphs=graph_dic)

    if args.method == 'rl':
        agent_class = DQAgent(graph_dic, args.model, args.lr, args.bs, args.n_step, args=args)
        if args.use_cuda:
            agent_class.cuda()
    elif args.method == 'random':
        agent_class = randomAgent()
    elif args.method == 'maxdegree': 
        agent_class = maxdegreeAgent()
    elif args.method == 'ada_greedy':
        agent_class = adaptive_greedyAgent()
    elif args.method == 'lazy_adaptive_greedy':
        agent_class = lazy_adaptive_greedyAgent()
    else: 
        assert(False)

    my_runner = runners.Runner(args, env_class, agent_class, args.verbose, logger=logger)

    if args.method == 'rl':
        final_reward = my_runner.loop()
    else:
        final_reward = my_runner.evaluate()
