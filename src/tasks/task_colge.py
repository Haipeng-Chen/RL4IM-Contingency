import sys
import argparse

import numpy as np
import networkx as nx

from src.runner import runners
from src.agent.colge.agent import Agent
from src.environment.graph import Graph
from src.environment.env import NetworkEnv, Environment
from src.environment.colge import env as colge_env
from src.tasks.task_basic_dqn import custom_env_parser, get_graph


def custom_arg_parse(parser: argparse.ArgumentParser):
    parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='MVC', help='Class to use for the environment. Must be in the \'environment\' module')
    parser.add_argument('--agent', metavar='AGENT_CLASS', default='Agent', type=str, help='Class to use for the agent. Must be in the \'agent\' module.')
    parser.add_argument('--graph_type', metavar='GRAPH', default='erdos_renyi', help ='Type of graph to optimize')
    parser.add_argument('--graph_nbr', type=int, default='1', help='number of differente graph to generate for the training sample')
    parser.add_argument('--model', type=str, default='S2V_QN_1', help='model name')
    parser.add_argument('--ngames', type=int, metavar='n', default=1, help='number of games to simulate')
    parser.add_argument('--niter', type=int, metavar='n', default='1000', help='max number of iterations per game')
    parser.add_argument('--epoch', type=int, metavar='nepoch', default=50, help="number of epochs")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--bs', type=int, default=32, help="minibatch size for training")
    parser.add_argument('--n_step',type=int, default=3, help="n step in RL")
    parser.add_argument('--node', type=int, metavar='nnode', default=20, help="number of node in generated graphs")
    parser.add_argument('--p', default=0.14, help="p, parameter in graph degree distribution")
    parser.add_argument('--m', default=4, help="m, parameter in graph degree distribution")
    parser.add_argument('--batch', type=int, metavar='nagent', default=None, help='batch run several agent at the same time')
    parser.add_argument('--verbose', action='store_true', default=True, help='Display cumulative results at each step')
    #-----------------------------------environment args-----------------------------------
    parser = custom_env_parser(parser)
    return parser.parse_args()


def run_colge(parser: argparse.ArgumentParser, run_args=None):
    args = custom_arg_parse(parser)
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
