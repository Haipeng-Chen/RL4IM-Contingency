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
from src.runner.utils import try_load_checkpint
import ipdb

def run_colge(_run, config, logger, run_args=None):
    args = SimpleNamespace(**config)
    graph_dic = {}
    #seed = 125
    #graph_one = graph.Graph(graph_type=args.graph_type, cur_n=20, p=0.15,m=4, seed=seed)
    print('Loading train graph: ', args.graph_type) 
    for graph_ in range(args.graph_nbr_train):
        #G, g, graph_name = get_graph(args.graph_index)
        #graph_dic[graph_] = Graph.create_graph(g)
        #graph_dic[graph_].graph_name = graph_name
        seed = graph_ #fix seed for now; TODO: revise later
        graph_dic[graph_]=Graph(graph_type=args.graph_type, cur_n=args.node_train, p=args.p,m=args.m,seed=seed)
        graph_dic[graph_].graph_name = str(graph_)
    print('train graphs in total: ', len(graph_dic))   

    #test graph
    print('Loading test graph: ', args.graph_type) 
    for graph_ in range(args.graph_nbr_train, args.graph_nbr_train+args.graph_nbr_test):
        seed = graph_
        graph_dic[graph_]=Graph(graph_type=args.graph_type, cur_n=args.node_test, p=args.p,m=args.m,seed=seed)
        graph_dic[graph_].graph_name = str(graph_)
    print('merged graphs length: ', len(graph_dic))

    env_class = Environment(cascade=args.cascade, T=args.T, budget=args.budget,
                           propagate_p=args.propagate_p, l=args.l, d=args.d, q=args.q, graphs=graph_dic)

    if args.method == 'rl':
        agent = DQAgent(graph_dic, args.model, args.lr, args.bs, args.n_step, args=args)
        if args.use_cuda:
            agent.cuda()
    elif args.method == 'random':
        agent = randomAgent()
    elif args.method == 'maxdegree': 
        agent = maxdegreeAgent()
    elif args.method == 'adaptive_greedy':
        agent = adaptive_greedyAgent()
    elif args.method == 'lazy_adaptive_greedy':
        agent = lazy_adaptive_greedyAgent()
    else: 
        assert(False)

    my_runner = runners.Runner(args, env_class, agent, args.verbose, logger=logger)
    try_load_checkpint(args=args, runner=my_runner, agent=agent)

    if args.mode == 'train':
        final_reward = my_runner.train()
    elif args.mode == 'test':
        final_reward = my_runner.evaluate()
    else:
        assert(False)
    
    #if args.method == 'rl':
    #    final_reward = my_runner.loop()
    #else:
    #    final_reward = my_runner.evaluate()
