import os
import time
import random
import logging

import torch
import numpy as np
import torch.nn.functional as F

import src.agent.colge.models as models
from src.agent.colge.utils.config import load_model_config


def epsilon_decay(init_v: float, final_v: float, step_t: int, decay_step: int):
    assert 0 < final_v <= 1, ValueError('Value Error')
    assert step_t >= 0, ValueError('Value Error')
    assert decay_step > 0, ValueError('Decay Value Error')
    
    if step_t >= decay_step:
        return final_v
    return step_t * ((final_v - init_v)/float(decay_step)) + init_v


class DQAgent:
    def __init__(self, graph, model, lr, bs, n_step, args=None):

        self.graphs = graph
        self.embed_dim = 64
        self.model_name = model

        self.k = 20
        self.alpha = 0.1
        self.gamma = 0.99
        self.lambd = 0.
        self.n_step=n_step
        self.args = args

        #self.epsilon_=1
        #self.epsilon_min=0.02
        #self.discount_factor =0.999990
        #self.eps_end=0.02
        #self.eps_start=1
        #self.eps_step=20000
        self.t=1
        self.memory = []
        self.memory_n=[]
        self.minibatch_length = bs

        if self.model_name == 'S2V_QN_1':
            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_1(**args_init)

        elif self.model_name == 'S2V_QN_2':
            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_2(**args_init)

        elif self.model_name== 'GCN_QN_1':
            args_init = load_model_config()[self.model_name]
            self.model = models.GCN_QN_1(**args_init)

        elif self.model_name == 'LINE_QN':
            args_init = load_model_config()[self.model_name]
            self.model = models.LINE_QN(**args_init)

        elif self.model_name == 'W2V_QN':
            args_init = load_model_config()[self.model_name]
            self.model = models.W2V_QN(G=self.graphs[self.games], **args_init)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.T = 5
        self.t = 1

        self.init_epsilon = 0.9
        self.final_epsilon = 0.01
        self.curr_epsilon = self.init_epsilon
        self.epislon_decay_steps = 500
        self.global_t = 0


    """
    p : embedding dimension
       
    """
    def reset(self, g):
        self.games = g
        if (len(self.memory_n) != 0) and (len(self.memory_n) % 300000 == 0):
            self.memory_n =random.sample(self.memory_n,120000)

        self.nodes = self.graphs[self.games].nodes()
        self.adj = self.graphs[self.games].adj()
        self.adj = self.adj.todense()
        self.adj = torch.from_numpy(np.expand_dims(self.adj.astype(int), axis=0))
        self.adj = self.adj.type(torch.FloatTensor)

        self.last_action = 0
        self.last_observation = torch.zeros(1, self.nodes, 1, dtype=torch.float)
        self.last_reward = -0.01
        self.last_done=0
        self.iter=1

        #self.init_epsilon = 1.
        #self.final_epsilon = 0.01
        #self.curr_epsilon = self.init_epsilon
        #self.epislon_decay_steps = 100
        #self.global_t = 0

    def act(self, observation, feasible_actions, mode):
        # to cuda
        if self.args.use_cuda:
            observation = observation.cuda()
            self.adj = self.adj.cuda()

        if self.curr_epsilon > np.random.rand() and mode != 'test':
            action = np.random.choice(feasible_actions)
        else:  # called for both test and train mode
            q_a = self.model(observation, self.adj)
            q_a = q_a.detach().cpu().numpy()
            
            # mask out unavailable action
            action_dim = q_a.shape[1]  # action_dim
            masked_out_action_id = list(set(range(action_dim)) - set(feasible_actions))
            if len(masked_out_action_id) > 0:
                q_a[0, masked_out_action_id, 0] = -9999999
            
            #action = np.where((q_a[0, :, 0] == np.max(q_a[0, :, 0][observation.cpu().numpy()[0, :, 0] == 0])))[0][0]
            action = int(np.argmax(q_a[0, :, 0]))

        if mode != 'test':
            # Update epsilon while training
            self.curr_epsilon = epsilon_decay(init_v=self.init_epsilon,
                                            final_v=self.final_epsilon,
                                            step_t=self.global_t,
                                            decay_step=self.epislon_decay_steps)
            self.global_t += 1
        
        return action

    def reward(self, observation, action, reward,done):
        loss = None
        if len(self.memory_n) > self.minibatch_length + self.n_step: #or self.games > 2:

            (last_observation_tens, action_tens, reward_tens, observation_tens, done_tens,adj_tens) = self.get_sample()
            # TODO further check
            target = self.to_cuda(reward_tens) + self.gamma *(1-self.to_cuda(done_tens)) * \
                torch.max(self.model(self.to_cuda(observation_tens), self.to_cuda(adj_tens)), dim=1)[0]
            target_f = self.model(self.to_cuda(last_observation_tens), self.to_cuda(adj_tens))
            target_p = target_f.clone()
            target_f[range(self.minibatch_length), action_tens, :] = target
            loss = self.criterion(target_p, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"[INFO] model update: t: {self.t}, loss: {loss}")

            #self.epsilon = self.eps_end + max(0., (self.eps_start- self.eps_end) * (self.eps_step - self.t) / self.eps_step)
            #if self.epsilon_ > self.epsilon_min:
               #self.epsilon_ *= self.discount_factor
        if self.iter>1:
            self.remember(self.last_observation, self.last_action, self.last_reward, observation.clone(),self.last_done*1)

        if done & self.iter> self.n_step:
              self.remember_n(False)
              new_observation = observation.clone()
              new_observation[:,action,:]=1
              self.remember(observation,action,reward,new_observation,done*1)

        if self.iter > self.n_step:
            self.remember_n(done)
        self.iter+=1
        self.t += 1
        self.last_action = action
        self.last_observation = observation.clone()
        self.last_reward = reward
        self.last_done = done
        return loss

    def get_sample(self):
        minibatch = random.sample(self.memory_n, self.minibatch_length - 1)
        minibatch.append(self.memory_n[-1])
        last_observation_tens = minibatch[0][0]
        action_tens = torch.Tensor([minibatch[0][1]]).type(torch.LongTensor)
        reward_tens = torch.Tensor([[minibatch[0][2]]])
        observation_tens = minibatch[0][3]
        done_tens =torch.Tensor([[minibatch[0][4]]])
        adj_tens = self.graphs[minibatch[0][5]].adj().todense()
        adj_tens = torch.from_numpy(np.expand_dims(adj_tens.astype(int), axis=0)).type(torch.FloatTensor)

        for last_observation_, action_, reward_, observation_, done_, games_ in minibatch[-self.minibatch_length + 1:]:
            last_observation_tens = torch.cat((last_observation_tens, last_observation_))
            action_tens = torch.cat((action_tens, torch.Tensor([action_]).type(torch.LongTensor)))
            reward_tens = torch.cat((reward_tens, torch.Tensor([[reward_]])))
            observation_tens = torch.cat((observation_tens, observation_))
            done_tens = torch.cat((done_tens,torch.Tensor([[done_]])))
            adj_ = self.graphs[games_].adj().todense()
            adj = torch.from_numpy(np.expand_dims(adj_.astype(int), axis=0)).type(torch.FloatTensor)
            adj_tens = torch.cat((adj_tens, adj))
        return (last_observation_tens, action_tens, reward_tens, observation_tens,done_tens, adj_tens)

    def remember(self, last_observation, last_action, last_reward, observation,done):
        self.memory.append((last_observation, last_action, last_reward, observation,done, self.games))

    def remember_n(self,done):
        if not done:
            step_init = self.memory[-self.n_step]
            cum_reward=step_init[2]
            for step in range(1,self.n_step):
                cum_reward+=self.memory[-step][2]
            self.memory_n.append((step_init[0], step_init[1], cum_reward, self.memory[-1][-3],self.memory[-1][-2], self.memory[-1][-1]))
        else:
            for i in range(1,self.n_step):
                step_init = self.memory[-self.n_step+i]
                cum_reward=step_init[2]
                for step in range(1,self.n_step-i):
                    cum_reward+=self.memory[-step][2]
                if i==self.n_step-1:
                    self.memory_n.append(
                        (step_init[0], step_init[1], cum_reward, self.memory[-1][-3], False, self.memory[-1][-1]))
                else:
                    self.memory_n.append((step_init[0], step_init[1], cum_reward,self.memory[-1][-3], False, self.memory[-1][-1]))

    def save_model(self, save_path):
        save_dir = os.path.join(save_path, str(self.global_t))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'model.pt'))

    def cuda(self):
        self.model.cuda()

    def to_cuda(self, tensor):
        if self.args.use_cuda:
            return tensor.cuda()
        else:
            return tensor

Agent = DQAgent
