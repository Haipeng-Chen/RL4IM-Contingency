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
import argparse

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
from baseline import *

#Q2: Use NaiveGCN for now, change it later




def is_fitted(sklearn_regressor):
    """Helper function to determine if a regression model from scikit-learn has
    ever been `fit`"""
    return hasattr(sklearn_regressor, 'n_outputs_')

class FQI(object):
    def __init__(self, graph, cascade='DIC', regressor=None):
        """Initialize simulator and regressor. Can optionally pass a custom
        `regressor` model (which must implement `fit` and `predict` -- you can
        use this to try different models like linear regression or NNs)"""
        self.env = NetworkEnv(G=graph, cascade=cascade)
        print('cascade model is: ', self.env.cascade)
        self.regressor = regressor or ExtraTreesRegressor()
    
    def state_action(self, state, action):
        #hp: it is a unique way of concatenating state and action, needs to use a more generalized way of representing it 
        #the input action is a list of nodes, needs to first convert it into a 1xN ndarray 
        #the output is a 3xN ndarray
        state=state.copy()
        action = action
        np_action = np.zeros((1, self.env.N))
        for n in action:
            np_action[0][n]=1
        state_action  = np.concatenate((state, np_action), axis=0)
        return state_action 
    
    #not used 
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
        possible_actions = self.env.possible_nodes
        if len(possible_actions)>int(self.env.budget):
            np.random.shuffle(possible_actions)
            Q_values = self.Q([state]*len(possible_actions), [[j] for j in possible_actions]) # enumerate all the possible nodes
            index=Q_values.argsort()[-int(self.env.budget):]
            next_action=[possible_actions[v] for v in index]
        else:
            next_action=np.array(possible_actions)
        return list(next_action)   
   
    def random_action(self, feasible_actions):
        #k is the number of chosen actions
        assert len(feasible_actions)>0
        action = random.choice(feasible_actions)
        #action = random.sample(feasible_actions,int(min(len(feasible_actions),self.env.budget)))
        return action

    def warm_start_action(self, feasible_actions):
        assert len(feasible_actions)>0
        #TODO: make it able to toogle
        action = max_degree(feasible_actions, self.env.G, self.env.budget) 
        return action   
 
    #not used
    def policy(self, state, eps=0.1):
        if np.random.rand() < eps:
            return self.random_action()
        else:
            return self.greedy_action(state) 
    
    #not used
    def run_episode(self, eps=0.1, discount=0.99):
        S, A, R = [], [], []
        cumulative_reward = 0
        self.env.reset()
        state = self.env.state
        for t in range(self.env.T):    
            state = self.env.state
            S.append(state)
            action = self.policy(state, eps)
            state_, reward=self.env.step(action=action)#Transition Happen 
            state=state_
            A.append(action)
            R.append(reward)
            cumulative_reward += reward * (discount**t)
        return S, A, R, cumulative_reward


    #not used
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

    #not used 
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
    #for primary agent done is for the last main step, for sec agent done is for last sub-step (node) in the last main step
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
    def __init__(self, graph, cascade='IC', lr_primary=0.001, lr_secondary=0.001):
        FQI.__init__(self, graph, cascade=cascade)
        self.feature_size = 3 
        self.net = NaiveGCN(node_feature_size=self.feature_size)
        self.net_list=[] #nets for secondary agents 
        for i in range(int(self.env.budget)): 
            self.net_list.append(NaiveGCN(node_feature_size=self.feature_size))
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr_primary)
        self.edge_index = torch.Tensor(list(nx.DiGraph(graph).edges())).long().t() #this is a 2 x num_edges tensor where each column is an edge
        self.loss_fn = nn.MSELoss()
        self.replay_memory = []
        self.optimizer_list=[]
        self.replay_memory_list = []
        for i in range(int(self.env.budget)):
            self.optimizer_list.append(optim.Adam(self.net_list[i].parameters(), lr=lr_secondary))
            self.replay_memory_list.append([])

        #shared scondary network ###########
        self.sec_net = NaiveGCN(node_feature_size=self.feature_size) ######
        self.sec_optimizer = optim.Adam(self.sec_net.parameters(), lr=lr_secondary) ########
        self.sec_replay_memory = [] ########
        self.memory_size = 1024

    def predict_rewards(self, state, action, netid='primary'): 
        #hp: split it into predict_rewards_primary and predict_rewards_secondary? when action becomes an embedding, they might be handled differently?
        #features = np.concatenate([[self.state_action(state, action)],[state]], axis=0).T #hp: revise later
        features = self.state_action(state, action).T
        #print('netid: ', netid)
        #print('action is: ', action)
        #print('feature dimension: ', features.shape)
        if netid == 'primary':
            net = self.net
        elif netid == 'secondary':
            net = self.sec_net
        else:
            net = self.net_list[netid]
        #net = self.net if netid == 'primary' else self.net_list[netid]
        graph_pred = net(torch.Tensor(features), self.edge_index) #.detach().numpy()
        return graph_pred

    #def batch_predict_rewards(self, states, actions, netid='primary'):
            
    def greedy_action_GCN(self, state, eps, eps_wstart=0):
        #series of action selection for secondary agents
        pri_action=[]
        sec_state = state
        possible_actions = self.env.feasible_actions.copy()
        #print('eps warm start is: ', eps_wstart)
        for i in range(int(self.env.budget)): 
            assert len(possible_actions) > 1
            chosen_sec_action = None
            p = np.random.rand()
            if p < eps: #[0, eps)
                #print('choosing random action')
                chosen_sec_action = self.random_action(possible_actions) 
            elif p < eps+eps_wstart: #[eps, eps+eps_wstart) #warm start
                print('choosing baseline action')
                chosen_sec_action = self.warm_start_action(possible_actions)
            else:
                #print('choosing RL action')
                #opt_sec_action = None
                max_reward = -1000
                for sec_action in possible_actions:
                    sec_action_ = [sec_action]
                    #sec_action_reward = self.predict_rewards(sec_state, sec_action_, netid=i)
                    sec_action_reward = self.predict_rewards(sec_state, sec_action_, netid='secondary')
                    if sec_action_reward > max_reward:
                        max_reward = sec_action_reward
                        chosen_sec_action = sec_action 
            pri_action.append(chosen_sec_action)
            possible_actions.remove(chosen_sec_action)
            sec_state[1][chosen_sec_action]=1 
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
                    #prediction = self.Q_GCN(state, action, netid= netid)
                    prediction = self.predict_rewards(state, action, netid= netid)
                    target = torch.tensor(reward, requires_grad=True)
                else:
                    next_state = memory.next_state.copy()
                    next_action = self.greedy_action_GCN(next_state, eps=0)
                    #prediction = self.Q_GCN(state, action, netid= netid) 
                    prediction = self.predict_rewards(state, action, netid= netid)
                    #target = reward + discount * self.Q_GCN(next_state, next_action, netid= netid) #hp: can revise to maintain a diff. net for each main step like secondary agents
                    target = reward + discount * self.predict_rewards(next_state, next_action, netid= netid)
                prediction_list.append(prediction.view(1))
                target_list.append(target.view(1))
                #loss = self.loss_fn(prediction, target) #TODO: revise to batch loss
                #print('netid is: ', netid)
                #print('state is: ', state)
                #print('action is: ', action)
                #print('Terminal state?', done)
                #print('Prediction: ', prediction.item())
                #print('Target: ', target.item())
                #print('one sample mse loss is: ', loss.item())
                #loss_list.append(loss)

        elif netid == 'secondary':
            for memory in batch_memory:
                state, action, reward, done = memory.state.copy(), memory.action.copy(), memory.reward, memory.done
                if done: #TODO: for secondary agents, it will only be done at the final node not the final step, revise it later
                    prediction = self.predict_rewards(state, action, netid= netid)
                    target = torch.tensor(float(reward), requires_grad=True) #the problem is some nodes at the final step will not add a TD term in target
                else:
                    next_state = memory.next_state.copy()
                    next_action = self.greedy_action_GCN(next_state, eps=0, eps_wstart=0)
                    prediction = self.predict_rewards(state, action, netid= netid)
                    target = reward + discount * self.predict_rewards(next_state, next_action, netid= netid)
                prediction_list.append(prediction.view(1))
                target_list.append(target.view(1))

        elif netid < self.env.budget-1:
            for memory in batch_memory:
                state, action, reward, done = memory.state.copy(), memory.action.copy(), memory.reward, memory.done
                next_state = memory.next_state.copy()
                next_action = self.greedy_action_GCN(next_state, eps=0, eps_wstart=0)
                #prediction = self.Q_GCN(state, action, netid= netid)
                prediction = self.predict_rewards(state, action, netid= netid)
                #next_prediction = self.Q_GCN(next_state, next_action, netid= netid+1)
                next_prediction = self.predict_rewards(state, action, netid= netid+1)
                #target = torch.tensor(float(reward), requires_grad=True) + discount * self.Q_GCN(next_state, next_action, netid= netid+1) 
                target = torch.tensor(float(reward), requires_grad=True) + discount * self.predict_rewards(next_state, next_action, netid= netid+1)
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
                    #prediction = self.Q_GCN(state, action, netid= netid)
                    prediction = self.predict_rewards(state, action, netid= netid)
                    target = torch.tensor(reward, requires_grad=True)
                else:
                    #when last secondary agent and not last main time step, update target using the first agent's (netid=0) Q network
                    next_state = memory.next_state.copy()
                    next_action = self.greedy_action_GCN(next_state, eps=0, eps_wstart=0)
                    #prediction = self.Q_GCN(state, action, netid= netid)
                    prediction = self.predict_rewards(state, action, netid= netid)
                    #target = reward + discount * self.Q_GCN(next_state, next_action, netid= 0) #hp: may also be updated using primary agent's Q network
                    target = reward + discount * self.predict_rewards(next_state, next_action, netid= 0)
                prediction_list.append(prediction.view(1))
                target_list.append(target.view(1))
                #loss = self.loss_fn(prediction, target)
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

    def fit_GCN(self, num_episodes=100, num_epochs=10, max_eps=0.3, min_eps=0.1, eps_decay=False, eps_wstart=0, batch_size = 16, discount=1, logdir=None):  
        if logdir == None:
            writer = SummaryWriter()
        else:
            writer = SummaryWriter(os.path.join('runs', logdir))
        best_value=0
        for epoch in range(num_epochs):#hp: remove epoch later? 
            loss_list = []
            cumulative_reward_list = [] #cumulative_reward_list: Reward assigned while training 
            true_cumulative_reward_list = [] #true_cumulative_reward_list: Reward assigned while testing 
            for episode in range(num_episodes):
                print('---------------------------------------------------------------')
                print('train episode: ', episode)
                if eps_decay:
                    eps=max(max_eps-0.005*episode, min_eps)
                else:
                    eps=min_eps
                S, A, R, NextS, D, cumulative_reward = self.run_episode_GCN(eps=eps,eps_wstart=eps_wstart, discount=discount)
                writer.add_scalar('cumulative reward', cumulative_reward, episode)
                #hp: the names of variables are misleading. Change it to primary-secondary
                new_memory = []
                new_memory_list=[]
                sec_new_memory = [] ########
                for _ in range(int(self.env.budget)):
                    new_memory_list.append([])
                horizon = self.env.T

                #----------------------------store memory---------------------------------
                for t in range(horizon):
                    #print('time step: ', t)
                    #print(S[t], A[t], R[t], NextS[t], D[t])
                    new_memory.append(Memory(S[t], A[t], R[t], NextS[t], D[t]))
                    #act=[]
                    sta = S[t].copy()
                    for i in range(int(self.env.budget)):
                        old_sta = sta.copy()
                        #rew=float(self.predict_rewards(sta, act, netid='primary')[0]) #this could leads to high bias; maybe try it later
                        done = False
                        if D[t] == False:  
                            rew = 0
                            sta[1][A[t][i]] = 1
                            next_sta = sta
                            done = False
                        elif i<self.env.budget-1:
                            rew = 0
                            sta[1][A[t][i]] = 1
                            next_sta = sta
                            done = False
                        else:
                            next_sta = None
                            rew = R[horizon-1]
                            done = True
                        sec_new_memory.append(Memory(old_sta, [A[t][i]], rew, next_sta, done))
                        #new_memory_list[i].append(Memory(old_sta, [A[t][i]], rew, next_sta, D[t])) 
                self.replay_memory += new_memory
                self.sec_replay_memory += sec_new_memory
                for i in range(int(self.env.budget)):
                    self.replay_memory_list[i]+=new_memory_list[i]

                #----------------------------update Q---------------------------------
                #hp: revise to update every time step
                '''############ removing update of primary net and multiple seconary nets
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
                for i in range(int(self.env.budget)):
                    if len(self.replay_memory_list[i]) >= batch_size:
                        batch_memory=self.replay_memory_list[i][-batch_size:].copy()
                        #batch_memory = np.random.choice(self.replay_memory_list[i], batch_size)
                        #print(batch_memory[0].state.shape, batch_memory[0].action)
                        self.optimizer_list[i].zero_grad()
                        loss = self.memory_loss(batch_memory,netid=i, discount=discount)
                        print('secondary {} loss is: {}'.format(i, loss.item()))
                        writer.add_scalar('secondary {} loss'.format(i), loss.item(), episode)
                        loss.backward()
                        self.optimizer_list[i].step()
                    if len(self.replay_memory_list[i]) > self.memory_size:
                        self.replay_memory_list[i] = self.replay_memory_list[i][-self.memory_size:]
                '''
                if len(self.sec_replay_memory) >= batch_size:
                    #batch_memory = np.random.choice(self.sec_replay_memory, batch_size)
                    batch_memory = self.sec_replay_memory[-batch_size:].copy()
                    self.sec_optimizer.zero_grad()
                    loss = self.memory_loss(batch_memory, netid='secondary', discount=discount)
                    print('secondary loss is: ', loss.item())
                    writer.add_scalar('secondary loss', loss.item(), episode)
                    loss.backward()
                    self.sec_optimizer.step()
                if len(self.sec_replay_memory) > self.memory_size:
                    self.sec_replay_memory = self.sec_replay_memory[-self.memory_size:]
                cumulative_reward_list.append(cumulative_reward)
            #print('Epoch {}, MSE loss: {}, average train reward: {}, discount test reward: {}'.format(epoch, np.mean(loss_list), np.mean(cumulative_reward_list), np.mean(true_cumulative_reward_list)))
        return cumulative_reward_list,true_cumulative_reward_list
    
    
    
    
    def run_episode_GCN(self, eps=0.1, eps_wstart=0, discount=0.99):
        S, A, R, NextS, D = [], [], [], [], []#D is for done -- indicator of terminal state
        cumulative_reward = 0
        a=0
        self.env.reset()
        for t in range(self.env.T): 
            state = self.env.state.copy()
            S.append(state)
            action = self.greedy_action_GCN(state, eps, eps_wstart)
            next_state, reward, done = self.env.step(action=action)#Transition Happen
            A.append(action)
            R.append(reward)
            NextS.append(next_state)
            D.append(done)
            cumulative_reward += reward * (discount**t)
        print('epsilon value in this episode is: ', eps)
        print('action in this episode is: ', A)
        print('episode total reward is: ', cumulative_reward)
        return S, A, R, NextS, D, cumulative_reward
    
def get_graph(graph_index):
    graph_list = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    graph_name = graph_list[graph_index]
    path = 'graph_data/' + graph_name + '.txt'
    G = nx.read_edgelist(path, nodetype=int)
    mapping = dict(zip(G.nodes(),range(len(G))))
    g = nx.relabel_nodes(G,mapping)
    return g, graph_name

def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments of influence maximzation')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32,
                help='batch size')
    parser.add_argument('--eps_decay', dest='eps_decay', type= bool, default=False, 
                help='is epsilon decaying?')
    parser.add_argument('--eps_wstart', dest='eps_wstart', type=float, default=0, 
                help='epsilon for warm start')
    parser.add_argument('--graph_index',dest='graph_index', type=int, default=2,
                help='graph index')
    parser.add_argument('--baseline',dest='baseline', type=str, default='ada_greedy',
                help='baseline')
    parser.add_argument('--cascade',dest='cascade', type=str, default='IC',
                help='cascade model')
    parser.add_argument('--greedy_sample_size',dest='greedy_sample_size', type=int, default=500,
                help='sample size for value estimation of greedy algorithms')
    parser.add_argument('--num_episodes', dest='num_episodes', type=int, default=100,
                help='number of training episodes')
    parser.add_argument('--logdir', dest='logdir', type=str, default=None, 
                help='log directory of tensorboard')

    #--------------------args rarely changed-------------------------------
    parser.add_argument('--discount', dest='discount', type=float, default=1.0, 
                help='discount factor')
    parser.add_argument('--First_time', dest='First_time', type=bool, default=True, 
                help='Is this the first time training?')
    #parser.add_argument(' ', dest=' ', type= , default= , 
                #help=' ')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    logdir = args.logdir
    batch_size = args.batch_size
    eps_decay = args.eps_decay
    eps_wstart = args.eps_wstart
    discount = args.discount
    First_time = args.First_time
    graph_index = args.graph_index
    cascade = args.cascade
    num_episodes = args.num_episodes
    g, graph_name=get_graph(graph_index)
    if First_time:
        model=DQN(graph=g, cascade=cascade)
        cumulative_reward_list,true_cumulative_reward_list=model.fit_GCN(num_episodes=num_episodes, num_epochs=1, max_eps=0.5, min_eps=0.1, 
                        discount=discount, eps_wstart=eps_wstart, logdir=logdir, batch_size=batch_size, eps_decay=eps_decay)
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
    for episode in range(10):
        print('---------------------------------------------------------------')
        print('test episode: ', episode)
        S, A, R, _, _, cumulative_reward = model.run_episode_GCN(eps=0, discount=discount)
        cumulative_rewards.append(cumulative_reward)
    print('average reward:', np.mean(cumulative_rewards))
    print('reward std:', np.std(cumulative_rewards))



