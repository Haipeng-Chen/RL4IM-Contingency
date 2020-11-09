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
from env import NetworkEnv


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
        #hp: seems to have warning
        #hp: it is a unique way of concatenating state and action, needs to use a more generalized way of representing it 
        #TODO: check Dai et al. 2017 
        output_state=states.copy()
        if len(actions)>0:
            #output_state[actions]=0
            output_state[actions]=1
        return output_state
    

    
    def Q(self, states, actions):
        states, actions = np.array(states), np.array(actions)
        if not is_fitted(self.regressor):
            return np.zeros(len(states))
        else:
            X = np.array([self.state_action(state , action ) for (state,action) in zip(states,actions)])
            y_pred = self.regressor.predict(X)
            return y_pred    

    #not used
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
    
    def run_episode(self, eps=0.1, discount=0.99):
        S, A, R = [], [], []
        cumulative_reward = 0
        self.simulator.reset()
        state = self.simulator.state
        for t in range(self.simulator.T):    
            state = self.simulator.state
            S.append(state)
            action = self.policy(state, eps)
            state_, reward=self.simulator.step(action=action)#Transition Happen 
            state=state_
            A.append(action)
            R.append(reward)
            cumulative_reward += reward * (discount**t)
        return S, A, R, cumulative_reward


    def fit_Q(self, episodes, num_iters=10, discount=0.99):
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

    
    def fit(self, num_refits=10, num_episodes=10, discount=0.99, eps=0.1):
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
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

class Memory_belief:
    def __init__(self, state):
        self.state = state


class DQN(FQI):
    def __init__(self, graph, lr_primary=0.001, lr_secondary=0.001):
        FQI.__init__(self, graph)
        self.feature_size = 2 #hp 
        self.net = NaiveGCN(node_feature_size=self.feature_size)
        self.net_list=[] #nets for secondary agents 
        for i in range(int(self.simulator.budget)): 
            self.net_list.append(NaiveGCN(node_feature_size=self.feature_size))
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr_primary)
        #print('graph edges: ', graph.edges())
        #print('graph edges: ', torch.Tensor(list(nx.DiGraph(graph).edges())))
        self.edge_index = torch.Tensor(list(nx.DiGraph(graph).edges())).long().t() #this is a 2 x num_edges tensor where each column is an edge
        #print('edge index is: ',self.edge_index.data.cpu().numpy())
        self.loss_fn = nn.MSELoss()
        self.replay_memory = []
        self.optimizer_list=[]
        self.replay_memory_list = []
        for i in range(int(self.simulator.budget)): 
            self.optimizer_list.append(optim.Adam(self.net_list[i].parameters(), lr=lr_secondary))
            self.replay_memory_list.append([])
        self.memory_size = 1024

    def predict_rewards(self, state, action, netid='primary'): # non backpropagatable
        #hp: split it into predict_rewards_primary and predict_rewards_secondary? when action becomes an embedding, they might be handled differently?
        features = np.concatenate([[self.state_action(state, action)],[state]], axis=0).T #hp: revise later
        net = self.net if netid == 'primary' else self.net_list[netid]
        graph_pred = net(torch.Tensor(features), self.edge_index) #.detach().numpy()
        #print('graph pred is: ', graph_pred)
        return graph_pred

    #def batch_predict_rewards(self, states, actions, netid='primary'):
            

    def Q_GCN(self, state, action, netid='primary'):
        #TODO: remove it later
        graph_pred = self.predict_rewards(state, action, netid)
        #y_pred = sum(node_pred[action]) # manually handle action selection
        return graph_pred

    def greedy_action_GCN(self, state):
        #series of action selection for secondary agents
        pri_action=[]
        sec_state = state
        possible_actions = self.simulator.feasible_actions.copy()
        #print('possible actions: ', possible_actions)
        for i in range(int(self.simulator.budget)): # greedy selection
            #node_rewards = self.predict_rewards(state, action, netid=i).reshape(-1)
            #action = [i] #hp: need to generalize
            if len(possible_actions)<2:#hp: what is 2 for? When there is only 1 candidate(posible infection) python will make possible_actions a element instead of list which makes strange things happen.
                possible_actions=self.simulator.all_nodes.copy() #hp: this should be revised
            #action_rewards = dict(zip(possible_actions, [None]*len(possible_actions)))
            max_reward = -1000
            opt_sec_action = None
            for sec_action in possible_actions:
                sec_action_ = [sec_action]
                sec_action_reward = self.predict_rewards(sec_state, sec_action_, netid=i)
                #print('reward for secondary action {} is {}: '.format(sec_action, sec_action_reward))
                if sec_action_reward > max_reward:
                    max_reward = sec_action_reward
                    opt_sec_action = sec_action 
            #node_rewards = self.predict_rewards(sec_state, action, netid=i).reshape(-1)
            #print('netid is: ', i)
            #print('secondary state is ', sec_state)
            #print('optimal secondary action is ', opt_sec_action)
            #print('predicted secondary max_reward: ', max_reward)
            #max_indice = node_rewards[possible_actions].argsort()[-1:]
            #node=np.array(possible_actions)[max_indice]
            pri_action.append(opt_sec_action)
            possible_actions.remove(opt_sec_action)
            sec_state[opt_sec_action]=1  #hp: wrong #another way is to define some state transtiion for secondary agent #TODO: revise it when state is trinary or other forms 
        return pri_action

    def memory_loss(self, batch_memory, netid='primary', discount=1):
        #hp: revise it to batch-based loss computation
        #hp: revise it to separate primary and secondary agents
        prediction_list = [] 
        target_list = []
        if netid == 'primary':
            for memory in batch_memory:
                state, action, reward, done = memory.state.copy(), memory.action.copy(), memory.reward, memory.done
                if done:
                    prediction = self.Q_GCN(state, action, netid= netid)
                    target = torch.tensor(reward, requires_grad=True)
                else:
                    next_state = memory.next_state.copy()
                    next_action = self.greedy_action_GCN(next_state)
                    prediction = self.Q_GCN(state, action, netid= netid) 
                    target = reward + discount * self.Q_GCN(next_state, next_action, netid= netid) #hp: can revise to maintain a diff. net for each main step like secondary agents
                prediction_list.append(prediction.view(1))
                target_list.append(target.view(1))
                #loss = self.loss_fn(prediction, target) #TODO: revise to batch loss
                '''
                if done:
                    print('netid is: ', netid)
                    #print('state is: ', state)
                    #print('action is: ', action)
                    print('Terminal state?', done)
                    print('Prediction: ', prediction.item())
                    print('Target: ', target.item())
                    print('one sample mse loss is: ', loss.item())
                '''
                #print('netid is: ', netid)
                #print('state is: ', state)
                #print('action is: ', action)
                #print('Terminal state?', done)
                #print('Prediction: ', prediction.item())
                #print('Target: ', target.item())
                #print('one sample mse loss is: ', loss.item())
                #loss_list.append(loss)
        elif netid < self.simulator.budget-1:
            for memory in batch_memory:
                state, action, reward, done = memory.state.copy(), memory.action.copy(), memory.reward, memory.done
                next_state = memory.next_state.copy()
                next_action = self.greedy_action_GCN(next_state)
                prediction = self.Q_GCN(state, action, netid= netid)
                next_prediction = self.Q_GCN(next_state, next_action, netid= netid+1)
                #target = reward + discount * self.Q_GCN(next_state, next_action, netid= netid+1) #this is problematic
                target = torch.tensor(float(reward), requires_grad=True) + discount * self.Q_GCN(next_state, next_action, netid= netid+1) 
                #loss = self.loss_fn(prediction, target)
                #print('netid is: ',netid)
                #print('Terminal state?', done) 
                #print('state is: ', state)
                #print('action is: ', action)
                #print('Prediction: ', prediction.item())
                #print('next prediction: ',next_prediction.item())
                #print('Target: ', target.item())
                #print('one sample mse loss is: ', loss.item())
                prediction_list.append(prediction.view(1))
                target_list.append(target.view(1))
                #loss_list.append(loss)
        else:
            for memory in batch_memory:
                state, action, reward, done = memory.state.copy(), memory.action.copy(), memory.reward, memory.done
                if done:
                    prediction = self.Q_GCN(state, action, netid= netid)
                    target = torch.tensor(reward, requires_grad=True)
                else:
                    #when last secondary agent and not last main time step, update target using the first agent's (netid=0) Q network
                    next_state = memory.next_state.copy()
                    next_action = self.greedy_action_GCN(next_state)
                    prediction = self.Q_GCN(state, action, netid= netid)
                    target = reward + discount * self.Q_GCN(next_state, next_action, netid= 0) #hp: may also be updated using primary agent's Q network
                prediction_list.append(prediction.view(1))
                target_list.append(target.view(1))
                #loss = self.loss_fn(prediction, target)
                '''
                if done:
                    print('netid is: ', netid)
                    #print('state is: ', state)
                    #print('action is: ', action)
                    print('Terminal state?', done)
                    print('Prediction: ', prediction.item())
                    print('Target: ', target.item())
                    print('one sample mse loss is: ', loss.item())
                '''
                #print('netid is: ',netid)
                #print('Terminal state?', done) 
                #print('state is: ', state)
                #print('action is: ', action)
                #print('Prediction: ', prediction.item())
                #print('Target: ', target.item())
                #print('one sample mse loss is: ', loss.item())
                #loss_list.append(loss) 
        batch_prediction = torch.stack(prediction_list)
        batch_target = torch.stack(target_list)
        batch_loss = self.loss_fn(batch_prediction, batch_target)
        #total_loss = sum(loss_list)
        #print(batch_loss)
        #return loss #hp: only loss is returned?
        return batch_loss

    def fit_GCN(self, num_episodes=100, num_epochs=10, max_eps=0.3, min_eps=0.1, eps_decay=False, batch_size = 16, discount=1, logdir=None):  
        if logdir == None:
            writer = SummaryWriter()
        else:
            writer = SummaryWriter(os.path.join('runs', logdir))
        best_value=0
        for epoch in range(num_epochs):#hp: why is epoch outside the episode loop? In this way you are basically running num_epochs*num_episodes episodes; see the DQN paper, they update q at every time step with a minibatch size of 32; let's keep it for now and revise later
            loss_list = []
            cumulative_reward_list = [] #cumulative_reward_list: Reward assigned while training (with discount factor=0.98)
            true_cumulative_reward_list = [] #true_cumulative_reward_list: Reward assigned while testing (with discount factor=1)
            true_RL_reward=[]# Objective function while testing, all 3 print for sanity check and can be remove.
            #self.simulator.Temperature=max((1-(epoch/20)),0) #hp: hard-coded, maybe revise later # This is for curiculum learning
            for episode in range(num_episodes):
                print('---------------------------------------------------------------')
                print('train episode: ', episode)
                if eps_decay:
                    eps=max(max_eps-0.005*episode, min_eps)
                else:
                    eps=min_eps
                S, A, R, NextS, D, cumulative_reward = self.run_episode_GCN(eps=eps, discount=discount)
                writer.add_scalar('primary reward', cumulative_reward, episode)
                #print(len(S),len(A))
                new_memory_belief=[]
                #hp: the names of variables are misleading. Change it to primary-secondary
                new_memory = []
                new_memory_list=[]
                for _ in range(int(self.simulator.budget)):
                    new_memory_list.append([])
                horizon = self.simulator.T
                for t in range(horizon):
                    #print('time step: ', t)
                    #print(S[t], A[t], R[t], NextS[t], D[t])
                    new_memory.append(Memory(S[t], A[t], R[t], NextS[t], D[t]))
                    act=[]
                    for i in range(int(self.simulator.budget)):
                        sta=self.state_action(S[t],act)
                        act.append(A[t][i])
                        #rew=float(self.predict_rewards(sta, act, netid='primary')[0]) #this could leads to high bias; maybe try it later
                        if D[t] == False: #t = horizon-1 
                            rew = 0
                        elif i<self.simulator.budget-1:
                            rew = 0
                        else:
                            rew = R[horizon-1]
  
                        new_memory_list[i].append(Memory(sta ,[np.array(A[t][i])],rew ,self.state_action(sta,act),D[t]) )
                self.replay_memory += new_memory
                for i in range(int(self.simulator.budget)):
                    self.replay_memory_list[i]+=new_memory_list[i]

                #----------------------------update Q---------------------------------
                #hp: revise to update every time step
                if len(self.replay_memory) >= batch_size:
                    #batch_memory = np.random.choice(self.replay_memory, batch_size)
                    batch_memory = self.replay_memory[-batch_size:].copy()
                    self.optimizer.zero_grad()
                    loss = self.memory_loss(batch_memory, discount=discount)
                    print('primary loss is: ', loss.item())
                    #loss_list.append(loss.item())
                    writer.add_scalar('primary loss', loss.item(), episode)
                    loss.backward()
                    self.optimizer.step()
                if len(self.replay_memory) > self.memory_size:
                    self.replay_memory = self.replay_memory[-self.memory_size:]
                for i in range(int(self.simulator.budget)):
                    if len(self.replay_memory_list[i]) >= batch_size:
                        batch_memory=self.replay_memory_list[i][-batch_size:].copy()
                        #batch_memory = np.random.choice(self.replay_memory_list[i], batch_size)
                        self.optimizer_list[i].zero_grad()
                        loss = self.memory_loss(batch_memory,netid=i, discount=discount)
                        print('secondary {} loss is: {}'.format(i, loss.item()))
                        writer.add_scalar('secondary {} loss'.format(i), loss.item(), episode)
                        loss.backward()
                        self.optimizer_list[i].step()
                    if len(self.replay_memory_list[i]) > self.memory_size:
                        self.replay_memory_list[i] = self.replay_memory_list[i][-self.memory_size:]
                cumulative_reward_list.append(cumulative_reward)
                #_, _, true_R, true_cumulative_reward = self.run_episode_GCN(eps=0, discount=1) #hp: what is this for?
                #true_RL_reward.append(sum(true_R))
                #true_cumulative_reward_list.append(true_cumulative_reward)
            #print('Epoch {}, MSE loss: {}, average train reward: {}, no discount test reward: {}, discount test reward: {}'.format(epoch, np.mean(loss_list), np.mean(cumulative_reward_list),np.mean(true_RL_reward), np.mean(true_cumulative_reward_list)))
        return cumulative_reward_list,true_cumulative_reward_list
    
    

    
    def run_episode_GCN(self, eps=0.1, discount=0.99):
        #hp: this should be majorly revised
        S, A, R, NextS, D = [], [], [], [], []#D is for done -- indicator of terminal state
        cumulative_reward = 0
        a=0
        self.simulator.reset()
        for t in range(self.simulator.T): 
            state = self.simulator.state.copy()
            #print('==========================')
            #print('state is ', state)
            S.append(state)
            action = self.policy_GCN(state, eps)
            next_state, reward, done = self.simulator.step(action=action)#Transition Happen
            A.append(action)
            R.append(reward)
            NextS.append(next_state)
            D.append(done)
            cumulative_reward += reward * (discount**t)
        print('epsilon value is: ', eps)
        print('action in this episode is: ', A)
        print('cumulated reward is: ', cumulative_reward)
        return S, A, R, NextS, D, cumulative_reward
    
    def policy_GCN(self, state, eps=0.1):
        s=state.copy()
        if np.random.rand() < eps:
            #print('taking random action')
            return self.random_action()
        else:
            #print('taking RL action')
            return self.greedy_action_GCN(s) 



def get_graph(graph_index):
    graph_list = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    graph_name = graph_list[graph_index]
    path = 'graph_data/' + graph_name + '.txt'
    G = nx.read_edgelist(path, nodetype=int)
    mapping = dict(zip(G.nodes(),range(len(G))))
    g = nx.relabel_nodes(G,mapping)
    return g, graph_name

if __name__ == '__main__':
    #logdir = 'eps-decay+last-16'
    logdir = None
    batch_size = 16 
    eps_decay = True 
    discount=1
    First_time=True
    graph_index=2
    g, graph_name=get_graph(graph_index)
    if First_time:
        model=DQN(graph=g)
        cumulative_reward_list,true_cumulative_reward_list=model.fit_GCN(num_episodes=100, num_epochs=1, discount=1, logdir=logdir, batch_size=batch_size, eps_decay=eps_decay)
        with open('Graph={}.pickle'.format(graph_name), 'wb') as f:
            pickle.dump([model,true_cumulative_reward_list], f)
    else:
        with open('Graph={}.pickle'.format(graph_name), 'rb') as f:
            X = pickle.load(f) 
        model=X[0]
        true_cumulative_reward_list=X[1]
    cumulative_rewards = []
    [print() for i in range(4)]
    print('testing')
    print(logdir)
    for episode in range(10):
        print('---------------------------------------------------------------')
        print('test episode: ', episode)
        #model.simulator.Temperature=0
        S, A, R, _, _, cumulative_reward = model.run_episode_GCN(eps=0, discount=discount)
        cumulative_rewards.append(cumulative_reward)
    print('average reward:', np.mean(cumulative_rewards))
    print('reward std:', np.std(cumulative_rewards))



