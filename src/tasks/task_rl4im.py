import os
import sys
import json
import time
import argparse

import numpy as np
import networkx as nx

from pathlib import Path
from types import SimpleNamespace

from src.runner import runners
from src.agent.rl4im.agent import DQAgent
from src.agent.baseline import *
from src.environment.graph import Graph
from src.environment.env import NetworkEnv, Environment
from src.runner.utils import try_load_checkpint
import ipdb


def load_grah(args):
    graph_dic = {}
    
    path = Path(os.path.dirname(os.path.realpath(__file__))).parent.parent
    if len(args.real_graph) != 0:
        #print('loading real graph')
        real_world_graphs = ["Exhibition", "Flu", "Hospital", "India", "irvine"]
        assert args.real_graph in real_world_graphs, f'{args.real_graph} not in the real graph list'

        args.graph_nbr_train = 1
        args.graph_nbr_test = 1

        for graph_ in range(args.graph_nbr_train):
            G = nx.read_edgelist(os.path.join(path, 'data', 'graph_data', args.real_graph + '.txt'), nodetype=int)
            graph_dic[graph_] = Graph(g=G, args=args)
            graph_dic[graph_].graph_name = str(args.real_graph)
        graph_dic[1] = graph_dic[0] 
    else:
        is_train = True
        print('Loading train graph: ', args.graph_type)
        for graph_ in range(args.graph_nbr_train):
            seed = graph_
            
            if args.sample_graph:
                G = nx.read_edgelist(os.path.join(path, 'data', 'graph_data', args.sample_graph_name + '.txt'), nodetype=int)
                graph_dic[graph_] = Graph(g=G, args=args, seed=seed)
            else:
                graph_dic[graph_] = Graph(graph_type=args.graph_type, cur_n=args.node_train, p=args.p, m=args.m, seed=seed, args=args, is_train=is_train)
    
            graph_dic[graph_].graph_name = str(graph_)
        print('train graphs in total: ', len(graph_dic))   

        #test graph
        print('Loading validation/test graph: ', args.graph_type)
        is_train = False
        for i, graph_ in enumerate(range(args.graph_nbr_train, args.graph_nbr_train+args.graph_nbr_test)):
            seed = 100000 + i if args.mode == 'train' else 200000 + i #if test then use another seed 
            
            if args.sample_graph:
                G = nx.read_edgelist(os.path.join(path, 'data', 'graph_data', args.sample_graph_name + '.txt'), nodetype=int)
                graph_dic[graph_] = Graph(g=G, args=args, seed=seed)
            else:
                graph_dic[graph_] = Graph(graph_type=args.graph_type, cur_n=args.node_test, p=args.p, m=args.m, seed=seed, args=args, is_train=is_train)
            
            graph_dic[graph_].graph_name = str(seed)
        print('merged graphs length: ', len(graph_dic))
    return graph_dic


def run_rl4im(_run, config, logger, run_args=None):
    args = SimpleNamespace(**config)
    graph_dic = load_grah(args)

    # save graph info
    with open(os.path.join(args.local_results_path, 'graph_info.json'), 'w') as f:
        node_num = {}
        for g_id, g in graph_dic.items():
            node_num[g.graph_name] = int(g.cur_n)
        json.dump(node_num, f, indent=4)

    # seed is changed in Graph and change back to the args.seed
    np.random.seed(args.seed)
    env_class = Environment(cascade=args.cascade, T=args.T, budget=args.budget,
                            propagate_p=args.propagate_p, q=args.q, graphs=graph_dic, args=args)

    if args.method == 'rl':
        agent = DQAgent(graph_dic, args.model, args.lr, args.bs, args.n_step, args=args)
        if args.use_cuda:
            agent.cuda()
    elif args.method == 'random':
        agent = None  
    elif args.method == 'greedy':
        agent = None  
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
        evaluate_start = time.time()
        final_reward = my_runner.evaluate()
        evaluate_end = time.time()
        print('runtime in seconds is: ', evaluate_end-evaluate_start)
    else:
        assert(False)
    
