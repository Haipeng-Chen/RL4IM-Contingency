import numpy as np
import networkx as nx
import math
import random
import argparse
import pulp
import os
from src.IC import runIC_repeat
from src.agent.baseline import *

import time


class Environment(object):
    '''
    Environment for influence maximization
    G is a nx graph
    state is 3xN binary array, 
    -- 1st row: invited in previous main step and came (=1), 
    -- 2nd row: invited but not come (=1); 
    -- 3rd row: invited in previous sub step (=1) or not (=0) --- only updated outside environment (in rl4im.py: greedy_action_GCN() and memory store step)
    '''
    def __init__(self, mode='train', T=20, budget=5, propagate_p = 0.1, l=0.05, d=1, q=1, cascade='IC', num_simul=1000, graphs=None, name='MVC', args=None):
        self.args = args
        self.name = name
        self.G = graphs[0] 
        self.graph_init = self.G  

        self.graphs = graphs
        self.mode = mode
        self.N = len(self.G.g)  
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
        self.state = np.zeros((3, self.N)) 
        self.observation = self.state
        nx.set_node_attributes(self.G.g, 0, 'attr')

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
                reward_max = reward_max/self.N*100 
                # reward_min
                seeds = fix_seeds + uncertain_seeds
                influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                seeds.append(sec_action)
                influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                reward_min = self.q*(influence_with - influece_without) 
                reward_min = reward_min/self.N*100  
                self.reward = (reward_max+reward_min)/2   
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
                reward_max = reward_max/self.N*100 
                # reward_min
                seeds = fix_seeds + uncertain_seeds
                influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                seeds.append(sec_action)
                influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                reward_min = self.q*(influence_with - influece_without)
                reward_min = reward_min/self.N*100 
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
        if cascade == 'IC':
            reward, _ = runIC_repeat(self.G.g, seeds, p=self.propagate_p, sample=sample)
        else:
            assert(False)
        return reward

    def f_multi(self, x):
        s=list(x) 
        val = self.run_cascade(seeds=s, cascade=self.cascade, sample=self.args.greedy_sample_size)
        return val
 
    #the simple state transition process
    def transition(self, invited):#q is probability being present
        present = []
        absent = []
        for i in invited:
            present.append(i) if random.random() <= self.q else absent.append(i)
        return present, absent

    def reset(self, g_index=0, mode='train'):
        self.mode = mode
        if mode == 'test': 
            self.G = self.graphs[g_index]
        else:
            self.G = self.graphs[g_index]
        self.N = len(self.G.g)
        self.A = nx.to_numpy_matrix(self.G.g)
        self.t = 0
        self.done = False
        self.reward = 0
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
