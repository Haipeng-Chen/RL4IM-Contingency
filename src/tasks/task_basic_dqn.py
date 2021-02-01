import os
import pdb
import sys
import time
import math
import random
import pickle
import itertools
import argparse, logging

import torch
import numpy as np
import networkx as nx
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from types import SimpleNamespace
from torch_geometric.nn import MessagePassing
from sklearn.ensemble import ExtraTreesRegressor
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv

from src.agent.dqn import DQN
from src.network.gcn import NaiveGCN
from src.environment.graph import Graph
from src.environment.env import NetworkEnv
#from src.agent.baseline import lazy_greedy, greedy, lazy_adaptive_greedy, adaptive_greedy, max_degree


def get_graph(graph_index):
    graph_list = ['test_graph', 'Hospital', 'India', 'Exhibition', 'Flu', 'irvine', 'Escorts', 'Epinions']
    graph_name = graph_list[graph_index]
    g = nx.read_edgelist(os.path.join('data', 'graph_data', graph_name + '.txt'), nodetype=int)
    G = Graph.create_graph(g)
    mapping = dict(zip(g.nodes(), range(len(g))))
    g = nx.relabel_nodes(G.g, mapping)
    return G, g, graph_name


def basic_dqn(_run, config, _log, run_args=None):
    args = SimpleNamespace(**config)
    
    logdir = args.logdir
    logfile = args.logfile
    First_time = args.First_time

    use_cuda = args.use_cuda
    memory_size = args.memory_size
    batch_size = args.batch_size
    batch_option = args.batch_option
    max_eps = args.max_eps
    min_eps = args.min_eps
    eps_decay = args.eps_decay
    eps_wstart = args.eps_wstart
    discount = args.discount
    cascade = args.cascade
    num_episodes = args.num_episodes
    greedy_sample_size = args.greedy_sample_size
    save_freq = args.save_freq

    graph_index = args.graph_index
    T = args.T
    #budget_ratio = args.budget_ratio
    budget = args.budget
    propagate_p = args.propagate_p
    l = args.l
    d = args.d
    q = args.q
    
    G, g, graph_name = get_graph(graph_index)
    print('chosen graph: ', graph_name)
    print('#nodes: ', len(g.nodes))
    print('#edges: ', len(g.edges))

    start_time = time.time()
    env = NetworkEnv(G=g, cascade=cascade, T=T, budget=budget, propagate_p=propagate_p, l=l, d=d, q=q)
    if First_time:
        model = DQN(graph=g, 
                    use_cuda=use_cuda, 
                    memory_size=memory_size, 
                    batch_size=batch_size, 
                    cascade=cascade, 
                    T=T, 
                    budget=budget, 
                    propagate_p=propagate_p, 
                    l=l, 
                    d=d, 
                    q=q, 
                    greedy_sample_size=greedy_sample_size, 
                    save_freq=save_freq,
                    env_config={
                        'env': env
                    })
        model.fit_GCN(batch_option=batch_option,
                      num_episodes=num_episodes,
                      max_eps=max_eps, min_eps=min_eps,
                      discount=discount,
                      eps_wstart=eps_wstart,
                      logdir=logdir,
                      graph_name=graph_name,
                      eps_decay=eps_decay)
        #with open('models/{}.pkl'.format(graph_name), 'wb') as f:
            #pickle.dump(model, f)
    else:
        with open('data/models/{}_{}.pkl'.format(graph_name, num_episodes-1), 'rb') as f:
            model = pickle.load(f)
    
    cumulative_rewards = []
    end_time = time.time()
    runtime = end_time - start_time
    print('-*-'*30)
    print('runtime for training process is: \n\n\n\n', runtime)

    print('testing')
    for episode in range(5):
        print('-*-'*30)
        print('test episode: ', episode)
        S, A, R, _, _, cumulative_reward = model.run_episode_GCN(eps=0, discount=discount)
        print('action in this episode is: ', A)
        print('episode total reward is: ', cumulative_reward)
        cumulative_rewards.append(cumulative_reward)
    print('-*-'*30)
    print('average reward:', np.mean(cumulative_rewards))
    print('reward std:', np.std(cumulative_rewards))
