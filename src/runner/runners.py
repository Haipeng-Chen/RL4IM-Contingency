import os
import time

import tqdm
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from src.utils.logging import Logger
from src.utils.os_utils import generate_id
import ipdb


class Runner:
    def __init__(self, args, environment, agent, verbose=True, logger: Logger=None):
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

    def evaluate(self, num_episode=2):
        """ Start evaluation """
        print('----------------------------------------------start evaluation---------------------------------------------------------')
        episode_accumulated_rewards = []
        feasible_actions = list(range(self.environment.N))
        mode = 'test'
        for episode in range(num_episode):
            self.environment.reset()
            self.agent.reset(0)  # g is zero
            
            accumulated_reward = 0
            for i in range(1, self.environment.T+1):
                state = self.environment.state.copy()
                if (i-1) % self.environment.budget == 0:
                    pri_action=[ ]
                sec_action = self.agent.act(th.from_numpy(state).float().transpose(1, 0)[None, ...], 
                                        feasible_actions=feasible_actions.copy(), mode=mode)
                print('feasible actions: ',feasible_actions)
                print('selected action: ', sec_action)
                feasible_actions = self.environment.try_remove_feasible_action(feasible_actions, sec_action)
                print('feasible actions: ',feasible_actions)
                pri_action.append(sec_action)
                next_state, reward, done = self.environment.step(i, pri_action, sec_action=sec_action)
                
                accumulated_reward += reward
                
                if done:
                    episode_accumulated_rewards.append(accumulated_reward)
                    print('accumulated reward of episode {} is: {}'.format(i, accumulated_reward))

        return np.mean(episode_accumulated_rewards)
    
    def loop(self, games, max_episodes):

        cumul_reward = 0.0
        list_cumul_reward=[]
        list_eval_reward=[]
        mode = 'train'
        st = time.time()

        for g in range(games):  # graph list
            print('game: {}'.format(g))
            for epoch in range(self.args.max_episodes):
                print('epoch: {}'.format(epoch))
                self.environment.reset()
                self.agent.reset(g)  # g is zero
                cumul_reward = 0.0
                pri_action = [ ]
                feasible_actions = list(range(self.environment.N))

                for i in range(1, self.environment.T+1):
                    state = self.environment.state.copy()
                    if (i-1) % self.environment.budget == 0:
                        pri_action=[ ]
                    sec_action = self.agent.act(th.from_numpy(state).float().transpose(1, 0)[None, ...], 
                                            feasible_actions=feasible_actions.copy(), mode=mode)

                    feasible_actions = self.environment.try_remove_feasible_action(feasible_actions, sec_action)
                    pri_action.append(sec_action)
                    next_state, reward, done = self.environment.step(i, pri_action, sec_action=sec_action)

                    # learning the model
                    loss = self.agent.reward(th.from_numpy(state).float().transpose(1, 0)[None, ...], sec_action, reward, done)
                    cumul_reward += reward
                    print(f"[INFO] Global_t: {self.agent.global_t}, Episode_t: {i}, Action: {sec_action}, Reward: {reward:.2f}, Epsilon: {self.agent.curr_epsilon:.2f}")
                    
                    # save the model
                    if (self.agent.global_t + 1) % 100 == 0:
                        self.agent.save_model(self.model_path)

                    if done:
                        print(f"[INFO] Global step: {self.agent.global_t}, Cumulative rewards: {cumul_reward}, Runtime (s): {(time.time()-st):.2f}")
                        print('--------------------------------------')
                        print(' ')
                        self.logger.log_stat(key='episode_reward', value=cumul_reward, t=self.agent.global_t)
                        if loss is not None:
                            self.logger.log_stat(key='loss', value=loss.detach().cpu().numpy(), t=self.agent.global_t)
                        
                        list_cumul_reward.append(cumul_reward)
                        break
                
                if (epoch + 1) % 5 == 0:
                    list_eval_reward.append(self.evaluate(num_episode=10))

            if self.verbose:
                print(" <=> Finished game number: {} <=>".format(g))
                print("")
        
        np.savetxt(os.path.join(self.results_path, 'train_episode_rewards.out'), list_cumul_reward, delimiter=',')
        np.savetxt(os.path.join(self.results_path, 'eval_episode_rewards.out'), list_eval_reward, delimiter=',')
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
