import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import pickle
import time
import random
import networkx as nx
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch.utils.tensorboard import SummaryWriter     

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesRegressor

from gcn import NaiveGCN
#from SIS_Belief_env import EpidemicEnv #hp: remove
from env import NetworkEnv


#Q1: did you use Memory_belief?
#Q2: Use NaiveGCN for now, change it later




def is_fitted(sklearn_regressor):
    """Helper function to determine if a regression model from scikit-learn has
    ever been `fit`"""
    return hasattr(sklearn_regressor, 'n_outputs_')

class FQI(object):
    def __init__(self, graph, regressor=None):
        """Initialize simulator and regressor. Can optionally pass a custom
        `regressor` model (which must implement `fit` and `predict` -- you can
        use this to try different models like linear regression or NNs)"""
        self.simulator = NetworkEnv(G=graph)
        self.regressor = regressor or ExtraTreesRegressor()
    
    def state_action(self, states,actions):
        output_state=states.copy()
        if len(actions)>0:
            output_state[actions]=0
            
        return output_state
    

    
    def Q(self, states, actions):
        states, actions = np.array(states), np.array(actions)
        if not is_fitted(self.regressor):
            return np.zeros(len(states))
        else:
            X = np.array([self.state_action(state , action ) for (state,action) in zip(states,actions)])
            y_pred = self.regressor.predict(X)
            return y_pred    
    
    def greedy_action(self, state):
        action = []
        possible_actions = self.simulator.possible_nodes
        if len(possible_actions)>int(self.simulator.budget):
            np.random.shuffle(possible_actions)
            Q_values = self.Q([state]*len(possible_actions), [[j] for j in possible_actions]) # enumerate all the possible nodes
            index=Q_values.argsort()[-int(self.simulator.budget):]
            next_action=[possible_actions[v] for v in index]
        else:
            next_action=np.array(possible_actions)
        return list(next_action)   
    
    def random_action(self):
        if len(self.simulator.feasible_actions)>0:
            action = random.sample(self.simulator.feasible_actions,int(min(len(self.simulator.feasible_actions),self.simulator.budget)))
        else:
            action = random.sample(self.simulator.all_nodes,int(min(self.simulator.n,self.simulator.budget)))
        return action
    
    def policy(self, state, eps=0.1):
        if np.random.rand() < eps:
            return self.random_action()
        else:
            return self.greedy_action(state) 
    
    def run_episode(self, eps=0.1, discount=0.98):
        S, A, R = [], [], []
        cumulative_reward = 0
        self.simulator.reset()
        state = self.simulator.state
        for t in range(self.simulator.T):    
            state = self.simulator.state
            S.append(state)
            action = self.policy(state, eps)
            state_, reward=self.simulator.step(action=action)#Transition Happen #hp: changed all perform to step
            state=state_
            A.append(action)
            R.append(reward)
            cumulative_reward += reward * (discount**t)
        return S, A, R, cumulative_reward


    def fit_Q(self, episodes, num_iters=10, discount=0.9):
        prev_S = []
        next_S = []
        rewards = []
        for (S, A, R) in episodes:
            horizon = len(S)
            for i in range(horizon-1):
                prev_S.append(list(self.state_action(S[i], A[i])) )
                rewards.append(R[i])
                next_S.append(S[i+1])
                

        prev_S = np.array(prev_S)
        next_S = np.array(next_S)
        rewards = np.array(rewards)

        for iteration in range(num_iters):
            best_actions = [self.greedy_action(state) for state in next_S]
            Q_best = self.Q(next_S, best_actions)
            y = list(rewards + discount * np.array(Q_best))
            self.regressor.fit(prev_S, y)

    
    def fit(self, num_refits=10, num_episodes=10, discount=0.9, eps=0.1):
        cumulative_rewards = np.zeros((num_refits, num_episodes))
        for refit_iter in range(num_refits):
            episodes = []
            for episode_iter in range(num_episodes):
                S, A, R, cumulative_reward = self.run_episode(eps=eps, discount=discount)
                cumulative_rewards[refit_iter,episode_iter] = cumulative_reward
                episodes.append((S, A, R))
            print('average reward:', np.mean(cumulative_rewards[refit_iter]))
            self.fit_Q(episodes,discount=discount)

        return episodes, cumulative_rewards    

class Memory:
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state

class Memory_belief:
    def __init__(self, state):
        self.state = state


class DQN(FQI):
    def __init__(self, graph, lr=0.005):
        FQI.__init__(self, graph)
        self.feature_size = 2 #hp: what are the 2? I can only imagine belief of the heath status which seems to be a scalar #oops the second feature was for the exp I test when assign the remaining budget as a feature. (ablation study)
        self.best_net = NaiveGCN(node_feature_size=self.feature_size)#hp: what is this for? Han Ching: I tried to store the net with best result on earlier exp, not used in this ver. I forgot to remove it. Change to 1 and remove the [state] in line 177 will be the original version.
        self.net = NaiveGCN(node_feature_size=self.feature_size)
        self.net_list=[] #nets for secondary agents 
        for i in range(int(self.simulator.budget)): 
            self.net_list.append(NaiveGCN(node_feature_size=self.feature_size))
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.edge_index = torch.Tensor(list(nx.DiGraph(graph).edges())).long().t()
        self.loss_fn = nn.MSELoss()
        self.replay_memory = []
        self.replay_memory_belief = [] #hp: never used? Han Ching yes could be remove
        self.memory_size = 5000

    def predict_rewards(self, state, action, netid='primary'): # non backpropagatable
        features = np.concatenate([[self.state_action(state, action)],[state]], axis=0).T
        net = self.net if netid == 'primary' else self.net_list[netid]
        return net(torch.Tensor(features), self.edge_index) #.detach().numpy()


    def Q_GCN(self, state, action, netid='primary'):
        node_pred = self.predict_rewards(state, action, netid)
        y_pred = sum(node_pred[action]) # manually handle action selection
        return y_pred

    def greedy_action_GCN(self, state):
        #series of action selection for secondary agents
        action=[]
        # possible_actions = self.simulator.possible_nodes.copy()
        possible_actions = self.simulator.feasible_actions.copy()
        for i in range(int(self.simulator.budget)): # greedy selection
            node_rewards = self.predict_rewards(state, action, netid=i).reshape(-1)
            if len(possible_actions)<2:#hp: what is 2 for? When there is only 1 candidate(posible infection) python will make possible_actions a element instead of list which makes strange things happen.
                possible_actions=self.simulator.all_nodes.copy()   
            max_indices = node_rewards[possible_actions].argsort()[-1:]
            node=np.array(possible_actions)[max_indices]
            action.append(node)
            possible_actions.remove(node)
            state[node]=0
        return action

    def memory_loss(self, batch_memory, discount=0.98):
        loss_list = []
        for memory in batch_memory:
            state, action, reward, next_state = memory.state.copy(), memory.action.copy(), memory.reward, memory.next_state.copy()
            next_action = self.greedy_action_GCN(next_state)
            prediction = self.Q_GCN(state, action)
            target = reward + discount * self.Q_GCN(next_state, next_action)
            loss = self.loss_fn(prediction, target)
            loss_list.append(loss)
        total_loss = sum(loss_list)
        return loss

    def fit_GCN(self, num_episodes=100, num_epochs=100, eps=0.1, discount=0.99): #hp: I changed discount from 0.9 to 0.99 as discount should be larger (or even 1) for us as time horizon is small (t=0, 1, 2, 3) 
        writer = SummaryWriter()
        best_value=0
        for epoch in range(num_epochs):#hp: why is epoch outside the episode loop? In this way you are basically running num_epochs*num_episodes episodes; see the DQN paper, they update q at every time step with a minibatch size of 32; let's keep it for now and revise later
            loss_list = []
            cumulative_reward_list = []#hp: what are the 3? #cumulative_reward_list: Reward assigned while training (with discount factor=0.98)
            true_cumulative_reward_list = [] #true_cumulative_reward_list: Reward assigned while testing (with discount factor=1)
            true_RL_reward=[]# Objective function while testing, all 3 print for sanity check and can be remove.
            #self.simulator.Temperature=max((1-(epoch/20)),0) #hp: hard-coded, maybe revise later # This is for curiculum learning
            for episode in range(num_episodes):
                S, A, R, cumulative_reward = self.run_episode_GCN(eps=eps, discount=discount)
                new_memory_belief=[]
                new_memory = []
                horizon = len(S) - 1
                for i in range(horizon):
                    new_memory.append(Memory(S[i], A[i], R[i], S[i+1]))
                self.replay_memory += new_memory
                batch_memory=self.replay_memory[-horizon:].copy()
                self.optimizer.zero_grad()
                loss = self.memory_loss(batch_memory, discount=discount)
                loss_list.append(loss.item())
                writer.add_scalar('\\Train/Loss\\', loss.item(), epoch)
                loss.backward()
                self.optimizer.step()
                if len(self.replay_memory) > self.memory_size:
                    self.replay_memory = self.replay_memory[-self.memory_size:]

                cumulative_reward_list.append(cumulative_reward)
                _, _, true_R, true_cumulative_reward = self.run_episode_GCN(eps=0, discount=1)
                true_RL_reward.append(sum(true_R))
                true_cumulative_reward_list.append(true_cumulative_reward)
            print('Epoch {}, MSE loss: {}, average train reward: {}, no discount test reward: {}, discount test reward: {}'.format(epoch, np.mean(loss_list), np.mean(cumulative_reward_list),np.mean(true_RL_reward), np.mean(true_cumulative_reward_list)))
        return cumulative_reward_list,true_cumulative_reward_list
    
    

    
    def run_episode_GCN(self, eps=0.1, discount=0.98):
        #hp: this should be majorly revised
        S, A, R = [], [], []
        cumulative_reward = 0
        a=0
        self.simulator.reset()
        for t in range(self.simulator.T): 
            # state = self.simulator.belief_state.copy() #hp: what is state? A list? array? # It's an np array
            state = self.simulator.state.copy()
            S.append(state)
            action = self.policy_GCN(state, eps)
            state_,reward=self.simulator.step(action=action)#Transition Happen
            A.append(action)
            R.append(reward)
            cumulative_reward += reward * (discount**t)
        return S, A, R, cumulative_reward
    
    def policy_GCN(self, state, eps=0.1):
        s=state.copy()
        if np.random.rand() < eps:
            return self.random_action()
        else:
            return self.greedy_action_GCN(s) 



def get_graph(graph_index):
    graph_list = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    graph_name = graph_list[graph_index]
    path = 'graph_data/' + graph_name + '.txt'
    G = nx.read_edgelist(path, nodetype=int)
    print(G.number_of_nodes())
    mapping = dict(zip(G.nodes(),range(len(G))))
    g = nx.relabel_nodes(G,mapping)
    return g, graph_name

if __name__ == '__main__':
    print('Here goes nothing')
    
    discount=1
    First_time=True
    graph_index=2
    g, graph_name=get_graph(graph_index)
    if First_time:
        model=DQN(graph=g)
        cumulative_reward_list,true_cumulative_reward_list=model.fit_GCN(num_episodes=10, num_epochs=30)#hp: I changed num_iterations to num_episodes
        with open('Graph={}.pickle'.format(graph_name), 'wb') as f:
            pickle.dump([model,true_cumulative_reward_list], f)
    else:
        with open('Graph={}.pickle'.format(graph_name), 'rb') as f:
            X = pickle.load(f) 
        model=X[0]
        true_cumulative_reward_list=X[1]
    cumulative_rewards = []
    for i in range(10):
        print('i is: ', i)
        model.simulator.Temperature=0
        S, A, R, cumulative_reward ,_ = model.run_episode_GCN(eps=0, discount=discount)
        cumulative_rewards.append(cumulative_reward)
    print('optimal reward:', np.mean(cumulative_rewards))
    print('optimal reward std:', np.std(cumulative_rewards))



