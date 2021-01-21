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
from torch_geometric.nn import MessagePassing
from sklearn.ensemble import ExtraTreesRegressor
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv

from src.agent.dqn import DQN
from src.network.gcn import NaiveGCN
from src.environment.graph import Graph
from src.environment.env import NetworkEnv
from src.agent.baseline import lazy_greedy, greedy, lazy_adaptive_greedy, adaptive_greedy, max_degree


class Memory:
    #for primary agent done is for the last main step, for sec agent done is for last sub-step (node) in the last main step
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


def get_graph(graph_index):
    graph_list = ['test_graph', 'Hospital', 'India', 'Exhibition', 'Flu', 'irvine', 'Escorts', 'Epinions']
    graph_name = graph_list[graph_index]
    g = nx.read_edgelist(os.path.join('data', 'graph_data', graph_name + '.txt'), nodetype=int)
    G = Graph.create_graph(g)
    mapping = dict(zip(G.nodes(), range(len(G))))
    g = nx.relabel_nodes(G.g, mapping)
    return G, g, graph_name


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments of influence maximzation')
    parser.add_argument('--use_cuda',dest='use_cuda', type=int, default=1,
                help='1 to use cuda 0 to not')
    parser.add_argument('--logfile', dest='logfile', type=str, default='logs/log',
                help='logfile for results ')
    parser.add_argument('--logdir', dest='logdir', type=str, default=None,
                help='log directory of tensorboard')
    parser.add_argument('--First_time', dest='First_time', type=int , default=1,
                help='Is this the first time training? 1 yes 0 no')
    
    #-----------------------------------hyper paras-----------------------------------
    parser.add_argument('--batch_option', dest='batch_option', type=str, default='random',
                help='option of batch sampling: random, last and mix')
    parser.add_argument('--memory_size', dest='memory_size', type=int, default=4096,
                help='replay memory size')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=128,
                help='batch size')
    parser.add_argument('--eps_decay', dest='eps_decay', type= int, default=0, 
                help='is epsilon decaying?')
    parser.add_argument('--eps_wstart', dest='eps_wstart', type=float, default=0, 
                help='epsilon for warm start')
    parser.add_argument('--ws_baseline',dest='ws_baseline', type=str, default='ada_greedy',
                help='baseline for warm_start')
    parser.add_argument('--num_episodes', dest='num_episodes', type=int, default=100,
                help='number of training episodes')
    parser.add_argument('--max_eps', dest='max_eps', type=float, default=0.3, 
                help='maximum probability for exploring random action')
    parser.add_argument('--min_eps', dest='min_eps', type=float, default=0.1, 
                help='minium probability for exploring random action')
    parser.add_argument('--discount', dest='discount', type=float, default=1.0,
                help='discount factor')
    parser.add_argument('--save_freq', dest='save_freq', type=int, default=100,
                help='frequency (in episodes) of saving models')

    #-----------------------------------environment args-----------------------------------
    parser.add_argument('--graph_index',dest='graph_index', type=int, default=2,
                help='graph index')
    parser.add_argument('--T', dest='T', type=int, default=4, 
                help='time horizon')
    #parser.add_argument('--budget_ratio', dest='budget_ratio', type=float, default=0.06, 
                #help='budget ratio; do the math: budget at each step = graph_size*budget_ratio/T')
    parser.add_argument('--budget', dest='budget', type=int, default=20,
                help='budget at each main step')
    parser.add_argument('--propagate_p', dest='propagate_p', type=float, default=0.1, 
                help='influence propagation probability')
    parser.add_argument('--l', dest='l', type=float, default=0.05,
                help='influence of each neighbor in LT cascade')
    parser.add_argument('--d', dest='d', type=float, default=1,
                help='d in SC cascade')
    parser.add_argument('--q', dest='q', type=float, default=1, 
                help='probability of invited node being present')
    parser.add_argument('--cascade',dest='cascade', type=str, default='IC',
                help='cascade model')
    parser.add_argument('--greedy_sample_size',dest='greedy_sample_size', type=int, default=500,
                help='sample size for value estimation of greedy algorithms')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
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
    if First_time:
        model = DQN(graph=g, use_cuda=use_cuda, memory_size=memory_size, batch_size=batch_size, cascade=cascade, T=T, budget=budget, propagate_p=propagate_p, l=l, d=d, q=q, greedy_sample_size=greedy_sample_size, save_freq=save_freq)
        model.fit_GCN(batch_option=batch_option, num_episodes=num_episodes,  max_eps=max_eps, min_eps=min_eps, 
                        discount=discount, eps_wstart=eps_wstart, logdir=logdir, eps_decay=eps_decay)
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
