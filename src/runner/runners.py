import os
import time
import json
import random

import tqdm
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from src.IC import celf
#TODO: move celf to agent.baselines

from src.utils.logging import Logger
from src.utils.os_utils import generate_id
import tracemalloc
import pdb


class Runner:
    def __init__(self, args, environment, agent, verbose=False, logger: Logger=None):
        self.environment = environment
        self.agent = agent
        self.verbose = verbose
        self.logger = logger
        self.args = args
        self.results_path = self.args.local_results_path
        os.makedirs(self.results_path, exist_ok=True)
        self.model_path = os.path.join(self.results_path, 'models')
        os.makedirs(self.model_path, exist_ok=True)

    def step(self):
        # observation = self.environment.observe().clone()
        # action = self.agent.act(observation).copy()
        # reward, done = self.environment.act(action)
        # self.agent.reward(observation, action, reward,done)
        # return observation, action, reward, done
        pass

    def state_abstraction(self, state):
        abs_state = state[0]+state[2]*self.args.q
        return abs_state

    def evaluate(self, num_episodes=10): #TODO
        """ Start evaluation """
        print(f'\n{"-"*50}start evaluation{"-"*50}')
        tracemalloc.start()

        episode_accumulated_rewards = np.empty((self.args.graph_nbr_test, num_episodes))
        feasible_actions = list(range(self.environment.N))
        mode = 'test'
        g_names = []
        
        #pbar = tqdm.tqdm(total=self.args.graph_nbr_test * num_episodes * self.environment.T)

        for g_index in range(self.args.graph_nbr_train, self.args.graph_nbr_train+self.args.graph_nbr_test):
            start_time = time.time()
            g_name = self.environment.graphs[g_index].graph_name
            g_names.append(g_name)
            print('graph: {}, nodes: {}, edges: {}'.format(g_name, len(self.environment.graphs[g_index].nodes), len(self.environment.graphs[g_index].edges)))
            if self.args.method == 'rl':
                for episode in range(num_episodes):
                    # select other graphs
                    self.environment.reset(g_index=g_index, mode=mode)
                    self.agent.reset(g_index=g_index, mode=mode)  
                    feasible_actions = list(range(self.environment.N))
                    accumulated_reward = 0
                    pri_action = [ ]
                    invited = []
                    presents = []

                    if self.args.verbose:
                        print('graph nodes: ', self.environment.graphs[g_index].g.nodes)
                        print('graph edge: ', self.environment.graphs[g_index].g.edges)

                    for i in range(1, self.environment.T+1):
                        state, state_padding, available_action_mask = self.environment.get_state(g_index)
                        if self.args.use_state_abs:
                            state = self.state_abstraction(state) 
                        sec_action = self.agent.act(state, feasible_actions=feasible_actions.copy(), mode=mode, mask=available_action_mask) 
    
                        if self.args.verbose:
                            print('current sub action: ', sec_action)

                        feasible_actions = self.environment.try_remove_feasible_action(feasible_actions, sec_action)
                        pri_action.append(sec_action)
                        _, _, done = self.environment.step(i, pri_action, sec_action=sec_action)

                        if i % self.environment.budget == 0:
                            present, _ = self.environment.transition(pri_action)
                            presents += present
                            invited += pri_action
                            pri_action=[ ]
                            
                            if self.args.verbose:
                                print('present: ', present)

                        if done:
                            accumulated_reward = self.environment.run_cascade(seeds=presents, cascade=self.environment.cascade, sample=self.args.num_simul_test)
                            episode_accumulated_rewards[g_index-self.args.graph_nbr_train, episode] = accumulated_reward / float(len(self.environment.graphs[g_index].nodes))
                            print('invited: ', invited)
                            print('present: ', presents) 
                        
                        #pbar.update(1)

            elif self.args.method == 'adaptive_greedy' or self.args.method == 'lazy_adaptive_greedy':
                print('method is :', self.args.method)
                for episode in range(num_episodes):
                    self.environment.reset(g_index=g_index, mode=mode)
                    feasible_actions = list(range(self.environment.N))
                    invited = []
                    presents = []
                    accumulated_reward = 0
                    for i in range(1, self.environment.T+1):
                        if (i-1) % self.environment.budget == 0:
                            #note that the other methods select budget number of nodes a time
                            pri_action, _ = self.agent.act(feasible_actions, self.environment.budget, self.environment.f_multi,presents)
                            invited+=pri_action
                            present, _ = self.environment.transition(pri_action)
                            presents+=present
                        for sec_action in pri_action:
                            feasible_actions = self.environment.try_remove_feasible_action(feasible_actions, sec_action)

                        if i == self.environment.T:
                            accumulated_reward = self.environment.run_cascade(seeds=presents, cascade=self.environment.cascade, sample=self.args.num_simul_test)
                            episode_accumulated_rewards[g_index-self.args.graph_nbr_train, episode] = accumulated_reward / float(len(self.environment.graphs[g_index].nodes))

                        #pbar.update(1)

            elif self.args.method == 'random':
                print('method is: ', self.args.method)
                for episode in range(num_episodes):
                    self.environment.reset(g_index=g_index, mode=mode)
                    feasible_actions = list(range(self.environment.N))
                    accumulated_reward = 0
                    invited = random.sample(feasible_actions, self.environment.T)
                    presents, _ = self.environment.transition(invited)
                    if self.args.verbose:
                        print('invited: ', invited)
                        print('present: ', presents)
                    accumulated_reward = self.environment.run_cascade(seeds=presents, cascade=self.environment.cascade, sample=self.args.num_simul_test)
                    episode_accumulated_rewards[g_index-self.args.graph_nbr_train, episode] = accumulated_reward / float(len(self.environment.graphs[g_index].nodes))
            elif self.args.method == 'greedy':
                print('method is: ', self.args.method)
                for episode in range(num_episodes):
                    self.environment.reset(g_index=g_index, mode=mode)
                    feasible_actions = list(range(self.environment.N))
                    accumulated_reward = 0
                    invited, _ = celf(self.environment.graphs[g_index].g, self.environment.T, p=self.environment.propagate_p, mc=10) 
                    presents, _ = self.environment.transition(invited)
                    accumulated_reward = self.environment.run_cascade(seeds=presents, cascade=self.environment.cascade, sample=self.args.num_simul_test)
                    episode_accumulated_rewards[g_index-self.args.graph_nbr_train, episode] = accumulated_reward / float(len(self.environment.graphs[g_index].nodes))
            end_time = time.time()
            print('runtime for one graph is: ', end_time-start_time)

            with open(os.path.join(self.results_path, 'test_mode_results.json'), 'w') as f:
                data = {
                    'g_names': g_names, 
                    'episode_accumulated_rewards': episode_accumulated_rewards.tolist(),
                }
                json.dump(data, f, indent=4)
        
        ave_cummulative_rewards = np.mean(episode_accumulated_rewards, axis=1)
        ave_cummulative_reward = np.mean(ave_cummulative_rewards)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()
        print('average cummulative reward vector is: ', ave_cummulative_rewards)
        print('average cummulative reward is: ', ave_cummulative_reward)
        print(f'{"-"*50}end evaluation{"-"*50}')
        print(' ')
        return g_names, episode_accumulated_rewards
    
    def train(self):

        cumul_reward = 0.0
        graph_cumul_reward = {}
        graph_eval_reward = {}
        mode = 'train'
        st = time.time()
        global_episode = 0
        terminate  = False

        #pbar = tqdm.tqdm(total=self.args.max_global_t)

        for epoch in range(self.args.nbr_epoch):
            print('epoch: ', epoch)
            if terminate:
                break
            for g_index in range(self.args.graph_nbr_train):  
                if terminate:
                    break
                                
                graph_name = self.agent.graphs[g_index].graph_name
                print('graph: {}, nodes: {}, edges: {}'.format(g_index, len(self.environment.graphs[g_index].nodes), len(self.environment.graphs[g_index].edges)))
                for episode in range(self.args.max_episodes):
                    if terminate:
                        break
                    global_episode += 1
                    self.environment.reset(g_index=g_index)
                    self.agent.reset(g_index)  
                    cumul_reward = 0.0
                    pri_action = [ ]
                    feasible_actions = list(range(self.environment.N))

                    curr_episode_t = 0
                    for i in range(1, self.environment.T+1):
                        
                        curr_episode_t += 1
                        
                        state, state_padding, available_action_mask = self.environment.get_state(g_index) #TODO: move it outside of environment
                        if self.args.use_state_abs:
                            state = self.state_abstraction(state) 
                            state_padding = self.state_abstraction(state_padding)
                        if (i-1) % self.environment.budget == 0:
                            pri_action=[ ]
                        sec_action = self.agent.act(state, feasible_actions=feasible_actions.copy(), mode=mode, mask=available_action_mask) 

                        feasible_actions = self.environment.try_remove_feasible_action(feasible_actions, sec_action)
                        pri_action.append(sec_action)
                        next_state, reward, done = self.environment.step(i, pri_action=pri_action, sec_action=sec_action, reward_type=self.args.reward_type) 

                        # learning the model
                        #loss = self.agent.reward(th.from_numpy(state).float().transpose(1, 0)[None, ...], sec_action, reward, done)
                        loss = self.agent.reward(state_padding, sec_action, reward, done, available_action_mask) ####
                        cumul_reward += reward
                        print(f"[INFO] Global_t: {self.agent.global_t}, Episode_t: {i}, Action: {sec_action}, Reward: {reward:.2f}, Loss: {loss}, Epsilon: {self.agent.curr_epsilon:.2f}")
                        
                        self.logger.log_stat(key=f'{graph_name}/epsilon', 
                                             value=self.agent.curr_epsilon, 
                                             t=self.agent.global_t)

                        if done:
                            # progress bar
                            #pbar.update(curr_episode_t)

                            print(f"\n[INFO] Global step: {self.agent.global_t}, Cumulative rewards: {cumul_reward}, Runtime (s): {(time.time()-st):.2f}")
                            print("-"*60)
                            print(' ')
                            self.logger.log_stat(key=f'{graph_name}/episode_reward', 
                                                 value=cumul_reward, 
                                                 t=self.agent.global_t)
                            if loss is not None:
                                self.logger.log_stat(key=f'{graph_name}/loss', 
                                                     value=loss.detach().item(), 
                                                     t=self.agent.global_t)
                            

                            if graph_name not in graph_cumul_reward:
                                graph_cumul_reward[graph_name] = [cumul_reward]
                            else:
                                graph_cumul_reward[graph_name].append(cumul_reward)

                            break    
                                    
                        if self.agent.global_t+1 >= self.args.max_global_t:
                            terminate = True
                            break 
                    if self.agent.global_t % self.args.save_every < self.environment.T:
                        print('saving the model')
                        self.agent.save_model(self.model_path)
                        g_names, episode_accumulated_rewards = self.evaluate()  #TODO: decrease samples 20->5; decrese validation inf samples 1000->100
                        mean_accumulated_reward_per_graph = np.mean(episode_accumulated_rewards, axis=1)
                        for i, graph_name in enumerate(g_names):
                            self.logger.log_stat(key=f'{graph_name}/eval_episode_reward',
                                                 value=mean_accumulated_reward_per_graph[i],
                                                 t=self.agent.global_t)
                            if graph_name not in graph_eval_reward:
                                graph_eval_reward[graph_name] = [episode_accumulated_rewards[i].tolist()]
                            else:
                                graph_eval_reward[graph_name].append(episode_accumulated_rewards[i].tolist())
                        
                # save per episode
                with open(os.path.join(self.results_path, 'train_episode_rewards.json'), 'w') as f:
                    json.dump(graph_cumul_reward, f, indent=4)
                    
                with open(os.path.join(self.results_path, 'eval_episode_rewards.json'), 'w') as f:
                    json.dump(graph_eval_reward, f, indent=4)
        
        return cumul_reward


def iter_or_loopcall(o, count):
    if callable(o):
        return [ o() for _ in range(count) ]
    else:
        # must be iterable
        return list(iter(o))


class BatchRunner:
    """
    Runs several instances of the same RL problem in parallel
    and aggregates the results.
    """

    def __init__(self, env_maker, agent_maker, count, verbose=False):
        self.environments = iter_or_loopcall(env_maker, count)
        self.agents = iter_or_loopcall(agent_maker, count)
        assert(len(self.agents) == len(self.environments))
        self.verbose = verbose
        self.ended = [ False for _ in self.environments ]

    def game(self, max_iter):
        rewards = []
        for (agent, env) in zip(self.agents, self.environments):
            env.reset()
            agent.reset()
            game_reward = 0
            for i in range(1, max_iter+1):
                observation = env.observe()
                action = agent.act(observation)
                (reward, stop) = env.act(action)
                agent.reward(observation, action, reward)
                game_reward += reward
                if stop :
                    break
            rewards.append(game_reward)
        return sum(rewards)/len(rewards)

    def loop(self, games,nb_epoch, max_iter):
        cum_avg_reward = 0.0
        for epoch in range(nb_epoch):
            for g in range(1, games+1):
                st_time = time.time()
                avg_reward = self.game(max_iter)
                cum_avg_reward += avg_reward
                ts = time.time() - st_time
                if self.verbose:
                    print("Simulation game {}:".format(g))
                    print(" ->            average reward: {}".format(avg_reward))
                    print(" -> cumulative average reward: {}".format(cum_avg_reward))
                    print(" -> expected total seconds: {:.2f}".format(ts * nb_epoch * games))
        return cum_avg_reward
