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


from src.network.gcn import NaiveGCN
from src.environment.env import NetworkEnv
from src.agent.baseline import *


def save_model(model, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


class Memory:
    #for primary agent done is for the last main step, for sec agent done is for last sub-step (node) in the last main step
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done


class DQN:
    def __init__(self, 
                 graph, 
                 use_cuda=1, 
                 cascade='IC',
                 memory_size=4096,
                 batch_size=128,
                 lr_primary=0.001,
                 lr_secondary=0.001,
                 T=4, budget=20,
                 propagate_p=0.1,
                 l=0.05,
                 d=1,
                 q=0.5,
                 greedy_sample_size=500,
                 save_freq=100, 
                 env_config=None):

        assert env_config is not None, ValueError('env_config should be a dict')
        
        self.env_config = env_config
        
        self.env = self.env_config['env']
        print('cascade model is: ', self.env.cascade)

        self.feature_size = 4 
        #self.net = NaiveGCN(node_feature_size=self.feature_size)
        #self.optimizer = optim.Adam(self.net.parameters(), lr=lr_primary)
        #self.replay_memory = []
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        print('device is: ', self.device)
        logging.info('device is: ', self.device)
        #shared scondary network 
        self.edge_index = torch.Tensor(list(nx.DiGraph(graph).edges())).long().t() #this is a 2 x num_edges tensor where each column is an edge
        self.edge_index = self.edge_index.to(self.device)
        self.loss_fn = nn.MSELoss()
        self.sec_net = NaiveGCN(node_feature_size=self.feature_size)
        self.sec_net.to(self.device)
        self.sec_optimizer = optim.Adam(self.sec_net.parameters(), lr=lr_secondary)
        self.sec_replay_memory = [] 
        self.memory_size = memory_size 
        self.batch_size = batch_size
        self.greedy_sample_size = greedy_sample_size
        self.save_freq = save_freq

    def state_action(self, state, action):
        #TODO: move it to DQN()
        #the input action is a list of nodes, needs to first convert it into a 1xN ndarray 
        #the output is a 4xN ndarray
        state=state.copy()
        action = action
        np_action = np.zeros((1, self.env.N))
        for n in action:
            np_action[0][n]=1
        state_action  = np.concatenate((state, np_action), axis=0)
        return state_action

    def predict_rewards(self, state, action, netid='primary'): 
        #TODO: enable batch prediction: the output could be an array whose dimension equals the number of feasible actions
        #hp: split it into predict_rewards_primary and predict_rewards_secondary? when action becomes an embedding, they might be handled differently?
        features = self.state_action(state, action).T
        if netid == 'primary':
            net = self.net
        elif netid == 'secondary':
            net = self.sec_net
        else:
            net = self.net_list[netid]
        #net = self.net if netid == 'primary' else self.net_list[netid]
        features = torch.Tensor(features).to(self.device) 
        graph_pred = net(features, self.edge_index) 
        return graph_pred

    #def batch_predict_rewards(self, states, actions, netid='primary'):

    def random_action(self, feasible_actions):
        #k is the number of chosen actions
        assert len(feasible_actions)>0
        action = random.choice(feasible_actions)
        #action = random.sample(feasible_actions,int(min(len(feasible_actions),self.env.budget)))
        return action

    def f_multi(self, x): #TODO: define f_multi() in baseline and pass greedy_sample_size to ?
        s=list(x)
        #print('cascade model is: ', env.cascade)
        val = self.env.run_cascade(seeds=s, cascade=self.env.cascade, sample=self.greedy_sample_size)
        return val

    def warm_start_actions(self, state, feasible_actions):
        #TODO: add other warm start methods such as max_betweeness etc
        assert len(feasible_actions)>0
        presents = [i for i in range(len(state[0])) if state[0][i]==1]#####could be used in printing in fit_GCN
        p = np.random.rand()
        if p<0:
            print('choosing max degree')
            action = max_degree(feasible_actions, self.env.G, self.env.budget) 
        elif p<0.01:
            print('choosing random')
            action = list(np.random.choice(feasible_actions, self.env.budget))
        else:
            print('choosing lazy_ada_greedy')
            action, _ =  adaptive_greedy(feasible_actions,self.env.budget,self.f_multi,presents)
        '''
        if p<0.25:
            print('choosing max degree')
            action = max_degree(feasible_actions, self.env.G, self.env.budget)
        elif p<0.5:
            print('choosing random')
            action = list(np.random.choice(feasible_actions, self.env.budget))
        else:
            print('choosing lazy_ada_greedy')
            action, _ =  lazy_adaptive_greedy(feasible_actions,self.env.budget,self.f_multi,presents)
        '''
        return action
            
    def policy(self, state, eps, eps_wstart=0):
        #series of action selection for secondary agents
        pri_action=[]
        sec_state = state.copy()
        feasible_actions = self.env.feasible_actions.copy()
        if len(self.sec_replay_memory) < self.batch_size:
            print('using warm-start actions')
            pri_action = self.warm_start_actions(sec_state, feasible_actions)
        elif np.random.rand() < eps_wstart: #warm start action
            assert len(feasible_actions) > 1
            pri_action = self.warm_start_actions(sec_state, feasible_actions)
        else:
            for i in range(int(self.env.budget)): 
                assert len(feasible_actions) > 1
                chosen_sec_action = None
                p = np.random.rand()
                if p < eps: 
                    chosen_sec_action = self.random_action(feasible_actions) 
                else:
                    #TODO: compress it using max etc; enable batch? 
                    max_reward = -1000
                    sec_action_rewards = [] #########
                    for sec_action in feasible_actions:
                        sec_action_ = [sec_action]
                        sec_action_reward = self.predict_rewards(sec_state, sec_action_, netid='secondary')
                        sec_action_rewards.append(sec_action_reward.item()) ########
                        if sec_action_reward > max_reward:
                            max_reward = sec_action_reward
                            chosen_sec_action = sec_action
                    #if eps==0 and eps_wstart==0 and i==0:##########
                        #print('state is: ', sec_state)
                        #print('sec reward for each node is: ', sec_action_rewards)
                pri_action.append(chosen_sec_action)
                feasible_actions.remove(chosen_sec_action)
                sec_state[2][chosen_sec_action]=1 
        return pri_action

    def memory_loss(self, batch_memory, netid='primary', discount=1):
        #hp: revise it to batch-based loss computation
        prediction_list = [] 
        target_list = []
        if netid == 'primary':
            for memory in batch_memory:
                state, action, reward, done = memory.state.copy(), memory.action.copy(), memory.reward, memory.done
                if done:
                    #prediction = self.Q_GCN(state, action, netid= netid)
                    prediction = self.predict_rewards(state, action, netid= netid)
                    target = torch.tensor(reward, requires_grad=True).to(self.device) 
                else:
                    next_state = memory.next_state.copy()
                    next_action = self.policy(next_state, eps=0) ####### this is wrong -- feasible_actions are diff. 
                    prediction = self.predict_rewards(state, action, netid= netid)   
                    target = reward + discount * self.predict_rewards(next_state, next_action, netid= netid)
                prediction_list.append(prediction.view(1))
                target_list.append(target.view(1))

        elif netid == 'secondary':
            for memory in batch_memory:
                state, action, reward, done = memory.state.copy(), memory.action.copy(), memory.reward, memory.done
                if done: 
                    prediction = self.predict_rewards(state, action, netid= netid)
                    target = torch.tensor(float(reward), requires_grad=True).to(self.device) 
                else:
                    next_state = memory.next_state.copy()
                    #get feasible_actions from next_state
                    m = np.sum(next_state, axis=0)
                    feasible_actions = [i for i in range(len(m)) if m[i]==0]
                    max_reward = -1000
                    #TODO: compress it like that in policy()
                    for next_action in feasible_actions: 
                        next_action_ = [action]
                        next_reward = self.predict_rewards(next_state, next_action_, netid='secondary')
                        if next_reward > max_reward:
                            max_reward = next_reward
                    prediction = self.predict_rewards(state, action, netid= netid)
                    target = reward + discount * max_reward
                prediction_list.append(prediction.view(1))
                target_list.append(target.view(1))
        else:
            assert(False)
        batch_prediction = torch.stack(prediction_list)
        batch_target = torch.stack(target_list)
        batch_loss = self.loss_fn(batch_prediction, batch_target)
        return batch_loss

    def fit_GCN(self,
                batch_option='random', 
                num_episodes=100, 
                max_eps=0.3, 
                min_eps=0.1, 
                eps_decay=False, 
                eps_wstart=0.1, 
                discount=1,
                graph_name='',
                logdir=None):
        if logdir == None:
            writer = SummaryWriter()
        else:
            writer = SummaryWriter(os.path.join('results', logdir))
        best_value=0
        loss_list = []
        #cumulative_reward_list = [] 
        #true_cumulative_reward_list = [] 
        for episode in tqdm(range(num_episodes)):
            print('---------------------------------------------------------------')
            print('train episode: ', episode)
            if eps_decay:
                eps = max(max_eps-0.005*episode, min_eps)
                eps_wstart = max(eps_wstart-0.005, 0)
            else:
                eps = min_eps
                eps_wstart = min_eps
            print('eps in this episode is: ', eps)
            print('eps_wstart in this episode is: ', eps_wstart)
            S, A, R, NextS, D, cumulative_reward = self.run_episode_GCN(eps=eps,eps_wstart=eps_wstart, discount=discount)
            writer.add_scalar('cumulative reward', cumulative_reward, episode)
            horizon = self.env.T
            presents = [i for i in range(len(S[horizon-1][0])) if S[horizon-1][0][i]==1] 
            print('action in this episode is: ', A)
            print('present: ', presents)
            print('episode total reward is: ', cumulative_reward)
            #new_memory = []
            sec_new_memory = [] 

            #----------------------------store memory---------------------------------
            for t in range(horizon):
                #new_memory.append(Memory(S[t], A[t], R[t], NextS[t], D[t]))
                sta = S[t].copy()
                for i in range(int(self.env.budget)):
                    st = time.time()
                    old_sta = sta.copy()
                    #rew=float(self.predict_rewards(sta, act, netid='primary')[0]) #this could leads to high bias; maybe try it later
                    done = False
                    if i<self.env.budget-1:
                        rew = 0
                        sta[2][A[t][i]] = 1
                        next_sta = sta.copy()
                        done = False
                    elif D[t] == False:
                        rew = 0
                        next_sta = S[t+1].copy()
                        done = False
                    else:
                        next_sta = None
                        rew = R[horizon-1]
                        done = True
                    sec_new_memory.append(Memory(old_sta, [A[t][i]], rew, next_sta, done))
                    
                    print(f'[INFO] Horizon: {t}/{horizon}, budget: {i}/{self.env.budget}, sec: {(time.time()-st):.2f}')

            self.sec_replay_memory += sec_new_memory

            #----------------------------update Q---------------------------------
            #hp: revise to update every time step
            '''
            #remove update of primary net 
            if len(self.replay_memory) >= self.batch_size:
                batch_memory = np.random.choice(self.replay_memory, self.batch_size)
                #batch_memory = self.replay_memory[-self.batch_size:].copy()
                self.optimizer.zero_grad()
                loss = self.memory_loss(batch_memory, discount=discount)
                print('primary loss is: ', loss.item())
                writer.add_scalar('primary loss', loss.item(), episode)
                loss.backward()
                self.optimizer.step()
            if len(self.replay_memory) > self.memory_size:
                self.replay_memory = self.replay_memory[-self.memory_size:]
            '''
            batch_option = batch_option 
            if len(self.sec_replay_memory) >= self.batch_size:
                if batch_option == 'random':
                    batch_memory = np.random.choice(self.sec_replay_memory, self.batch_size) 
                elif batch_option == 'last':
                    batch_memory = self.sec_replay_memory[-self.batch_size:].copy()
                elif batch_option == 'mix':
                    batch_memory = np.random.choice(self.sec_replay_memory, self.batch_size) if np.random.rand() < 0.5 else self.sec_replay_memory[-self.batch_size:].copy() 
                else:
                    assert(False)
                self.sec_optimizer.zero_grad()
                loss = self.memory_loss(batch_memory, netid='secondary', discount=discount)
                print('secondary loss is: ', loss.item())
                writer.add_scalar('secondary loss', loss.item(), episode)
                loss.backward()
                self.sec_optimizer.step()
                torch.cuda.empty_cache()
            
            if len(self.sec_replay_memory) > self.memory_size:
                self.sec_replay_memory = self.sec_replay_memory[-self.memory_size:]
            
            if (episode+1) % self.save_freq == 0:
                save_model(self, 'models/{}_{}.pkl'.format(graph_name, episode))

    def run_episode_GCN(self, eps=0.1, eps_wstart=0, discount=0.99):
        S, A, R, NextS, D = [], [], [], [], []
        cumulative_reward = 0
        a=0
        self.env.reset()
        for t in tqdm(range(self.env.T)):
            st = time.time()
            state = self.env.state.copy()
            S.append(state)
            action = self.policy(state, eps, eps_wstart)
            next_state, reward, done = self.env.step(action=action)
            A.append(action)
            R.append(reward)
            NextS.append(next_state)
            D.append(done)
            cumulative_reward += reward * (discount**t)
            print(f">>>> Current Episode: A: {A}, R: {R}, Sec: {(time.time()-st):.2f}")
        return S, A, R, NextS, D, cumulative_reward
