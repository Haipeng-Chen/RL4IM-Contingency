import numpy as np
import networkx as nx
import math
import random
import argparse
import pulp
#from influence import influence, parallel_influence
import os
from src.IC import runIC_repeat
from src.IC import runDIC_repeat
from src.IC import runLT_repeat
from src.IC import runSC_repeat
from src.agent.baseline import *

import time
import pdb

class NetworkEnv(object):
    '''
    Environment for peer leader selection process of influence maximization
    
    we consider fully known and static graph first, will potentially extend to: 1) dynamic graph 2) unknown graph 3) influence at each step 4) ...
    G is a nx graph
    node 'attr': a trinary value where 0 is not-selected; 1 is selected and present; 2 is selected but not present; I am planning to put the state and action embedding outside environment 
    state is 3xN binary array, 
    -- 1st row: invited in previous main step and came (=1), 
    -- 2nd row: invited but not come (=1); 
    -- 3rd row: invited in previous sub step (=1) or not (=0) --- it is only useful in states in the sub steps, not updated in env  
    -- elements (=0) on both 1st and 2nd rows are not invited and thus are feasible actions
    note that the 3rd row of state is only updated outside environment (in rl4im.py: greedy_action_GCN() and memory store step)
    '''
    
    def __init__(self, mode='train', T=20, budget=5, propagate_p = 0.1, l=0.05, d=1, q=1, cascade='IC', num_simul=1000, graphs=None):
        self.graphs = graphs
        self.mode = mode

    def step(self, i, pri_action, sec_action, reward_type=0):
        '''
        pri_action is a list, sec_action is an int

        reward type categories, example seed nodes before {1, 2, 3}, new node x
        0: reward0 = f({1, 2, 3, x}) - f({1, 2, 3})
        1: reward1 = f({x}) - f({ })
        2: reward2 = (reward0+reward1)/2
        3: use probabilty q 
        '''

        #compute reward as marginal contribution of a node
        if self.mode == 'train':
            if reward_type == 0:
                seeds = []
                [seeds.append(v) for v in range(self.N) if (self.state[0][v]==1 or self.state[2][v]==1)] 
                influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                seeds.append(sec_action)
                influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                self.reward = self.q*(influence_with - influece_without)
                self.reward = self.reward/self.N*100   ####
            elif reward_type == 1:
                seeds = []
                [seeds.append(v) for v in range(self.N) if self.state[0][v]==1] 
                influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                seeds.append(sec_action)
                influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                self.reward = self.q*(influence_with - influece_without) 
                self.reward = self.reward/self.N*100  ###
            elif reward_type == 2:
                fix_seeds = []
                [fix_seeds.append(v) for v in range(self.N) if self.state[0][v]==1]
                uncertain_seeds = []
                [uncertain_seeds.append(v) for v in range(self.N) if self.state[2][v]==1]
                # reward_max
                seeds = fix_seeds.copy()
                influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                seeds.append(sec_action)
                influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                reward_max = self.q*(influence_with - influece_without) 
                reward_max = reward_max/self.N*100 #####
                # reward_min
                seeds = fix_seeds + uncertain_seeds
                influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                seeds.append(sec_action)
                influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                reward_min = self.q*(influence_with - influece_without) 
                reward_min = reward_min/self.N*100  ####
                self.reward = (reward_max+reward_min)/2   ####
            elif reward_type == 3:
                fix_seeds = []
                [fix_seeds.append(v) for v in range(self.N) if self.state[0][v]==1]
                uncertain_seeds = []
                [uncertain_seeds.append(v) for v in range(self.N) if self.state[2][v]==1]
                # reward_max
                seeds = fix_seeds.copy()
                influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                seeds.append(sec_action)
                influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                reward_max = self.q*(influence_with - influece_without)
                reward_max = reward_max/self.N*100 ####
                # reward_min
                seeds = fix_seeds + uncertain_seeds
                influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                seeds.append(sec_action)
                influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                reward_min = self.q*(influence_with - influece_without)
                reward_min = reward_min/self.N*100 ####
                self.reward = reward_min*self.q+reward_max*(1-self.q)
            else:
                assert(False)

        #update next_state and done      
        if i%self.budget == 0:
        #a primary step
            invited = pri_action
            present, absent = self.transition(invited)
            state=self.state.copy()
            for v in present:
                self.state[0][v]=1
            for v in absent:
                self.state[1][v]=1
            self.state[2].fill(0)
            if i == self.T:
                next_state = None
                self.done = True
            else:
                next_state = self.state.copy()
                self.done = False
        else:
        #a secondary step
            self.state[2][sec_action]=1
            next_state = self.state.copy()
            self.done = False

        if i == self.T:  
            next_state = None
            self.done = True

        return next_state, self.reward, self.done
            
    def run_cascade(self, seeds, cascade='IC', sample=1000):
        #there may be better ways of passing the arguments
        if cascade == 'IC':
            reward, _ = runIC_repeat(self.G.g, seeds, p=self.propagate_p, sample=sample)
        elif cascade == 'DIC':
            reward, _ = runDIC_repeat(self.G.g, seeds, p=self.propagate_p, q=0.001, sample=sample)
        elif cascade == 'LT':
            reward, _ = runLT_repeat(self.G.g, seeds, l=self.l, sample=sample)
        elif cascade == 'SC':
            reward, _ = runSC_repeat(self.G.g, seeds, d=self.d, sample=sample)
        else:
            assert(False)
        return reward

    #TODO
    def f_multi(self, x):
        s=list(x) 
        #print('cascade model is: ', env.cascade)
        val = self.run_cascade(seeds=s, cascade=self.cascade, sample=self.args.greedy_sample_size)
        return val
 
    #the simple state transition process
    def transition(self, invited):#q is probability being present
        present = []
        absent = []
        for i in invited:
            present.append(i) if random.random() <= self.q else absent.append(i)
        #[present.append(i) for i in invited if random.random() <= q]
        return present, absent

    def reset(self, g_index=0, mode='train'):
        self.mode = mode
        if mode == 'test': 
            #self.graph_index = g_index 
            self.G = self.graphs[g_index]
        else:
            #self.graph_index = g_index
            self.G = self.graphs[g_index]
        self.N = len(self.G.g)
        self.A = nx.to_numpy_matrix(self.G.g)
        self.t = 0
        self.done = False
        self.reward = 0
        self.state = np.zeros((3, self.N)) 
        self.observation = self.state
        nx.set_node_attributes(self.G.g, 0, 'attr')


class Environment(NetworkEnv):
    def __init__(self, T=20, budget=5, propagate_p = 0.1, l=0.05, d=1, q=1, cascade='IC', num_simul=1000, graphs=None, name='MVC', args=None):
        super().__init__(T=T,
                         budget=budget,
                         propagate_p=propagate_p,
                         l=l,
                         d=d,
                         q=q,
                         cascade=cascade,
                         num_simul=num_simul,
                         graphs=graphs)
        self.args = args
        self.name = name
        self.G = graphs[0] ####
        self.graph_init = self.G  #####

        self.graphs = graphs
        self.N = len(self.G.g)  ####
        #self.budget = math.floor(self.N * budget_ratio/T)
        self.budget = budget
        self.A = nx.to_numpy_matrix(self.G.g)  
        self.propagate_p = propagate_p
        self.l = l
        self.d = d
        self.q = q
        self.T = T
        self.cascade = cascade
        self.num_simul = self.args.num_simul_train
        self.t = 0
        self.done = False
        self.reward = 0
        #self.feasible_actions = list(range(self.N))
        self.state = np.zeros((3, self.N)) 
        self.observation = self.state
        nx.set_node_attributes(self.G.g, 0, 'attr')

    def get_state(self, g_index):
        curr_g = self.graphs[g_index]
        available_action_mask = np.array([1] * curr_g.cur_n + [0] * (curr_g.max_node_num - curr_g.cur_n))

        # padding the state for storing
        obs_padding = self.observation.copy()
        if self.args.model_scheme == 'type1':
            padding = np.repeat(np.array([-1] * (curr_g.max_node_num - curr_g.cur_n))[None, ...], self.observation.shape[0], axis=0)
            obs_padding = np.concatenate((self.observation.copy(), padding), axis=-1)
        return self.observation.copy(), obs_padding, available_action_mask

    def try_remove_feasible_action(self, feasible_actions, sec_action):
        try:
            feasible_actions.remove(sec_action)
            return feasible_actions
        except Exception:
            pass
        finally:
            return feasible_actions


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments of influence maximzation')
    parser.add_argument('--baseline',dest='baseline', type=str, default='ada_greedy',
                help='baseline, could be ada_greedy, random, maxdegree')
    parser.add_argument('--graph_index',dest='graph_index', type=int, default=2,
                help='graph index')
    parser.add_argument('--T', dest='T', type=int, default=1,
                help='time horizon')
    #parser.add_argument('--budget_ratio', dest='budget_ratio', type=float, default=0.06,
                #help='budget ratio; do the math: budget at each step = graph_size*budget_ratio/T')
    parser.add_argument('--budget', dest='budget', type=int, default=20,
                help='budget at each main step')

    parser.add_argument('--cascade',dest='cascade', type=str, default='IC',
                help='cascade model')
    parser.add_argument('--propagate_p', dest='propagate_p', type=float, default=0.1,
                help='influence propagation probability')
    parser.add_argument('--l', dest='l', type=float, default=0.05,
                help='influence of each neighbor in LT cascade')
    parser.add_argument('--d', dest='d', type=float, default=1,
                help='d in SC cascade')
    parser.add_argument('--q', dest='q', type=float, default=1,
                help='probability of invited node being present')
    parser.add_argument('--num_simul',dest='num_simul', type=int, default=1000,
                help='number of simulations for env.step')
    parser.add_argument('--greedy_sample_size',dest='greedy_sample_size', type=int, default=500,
                help='sample size for value estimation of greedy algorithms')

    return parser.parse_args()

