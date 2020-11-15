import numpy as np
import networkx as nx
import math
import random
#from influence import influence, parallel_influence
from IC import runIC_repeat

class NetworkEnv(object):
    '''
    Environment for peer leader selection process of influence maximization
    
    we consider fully known and static graph first, will potentially extend to: 1) dynamic graph 2) unknown graph 3) influence at each step 4) ...

    G is a nx graph

    node 'attr': a trinary value where 0 is not-selected; 1 is selected and present; 2 is selected but not present; I am planning to put the state and action embedding outside environment #hp: removed this

    state is a 2xN binary ndarray, the first row means invited in previous main step and came (=1) or invited but not come (=0) or not invited (=0), second row means invited in previous sub step (=1) or not (=0)  
    note that the 2nd row of state will also be updated outside environment (in greedy_action_GCN())
    '''
    
    def __init__(self, G, T=4, budget_ratio=0.06, propagate_p = 0.1, q=0.6):
        self.G = G
        self.N = len(self.G)
        self.budget = math.floor(self.N * budget_ratio/T)
        self.A = nx.to_numpy_matrix(self.G)  
        self.propagate_p = propagate_p
        self.q = q
        self.T = T
        self.t = 0
        self.done = False
        self.reward = 0
        self.feasible_actions = list(range(self.N))
        self.state=np.zeros((2, self.N)) 
        #print('initialized state: ',self.state)
        nx.set_node_attributes(self.G, 0, 'attr')

    def step(self, action):
        invited = action
        present, absent = self.transition(invited)
        state=self.state.copy()
        for v in present:
            self.G.nodes[v]['attr']=1
            self.state[0][v]=1
            self.state[1][v]=0
        for v in absent:
            self.G.nodes[v]['attr']=2
            self.state[0][v]=0
            self.state[1][v]=0
        if self.t == self.T-1:
            seeds = []
            [seeds.append(v) for v in range(self.N) if self.G.nodes[v]['attr'] == 1]
            self.reward, _ = runIC_repeat(self.G, seeds, p=self.propagate_p, sample=1000)
            next_state = None
            self.done = True
        else:
            self.reward = 0 #TODO: add an auxilliary reward to warm-start 
            #Han Ching: One idea is to simulate the IM here with given seend.
            next_state = self.state.copy()
            feasible_actions_cp = self.feasible_actions.copy()
            self.feasible_actions = [i for i in feasible_actions_cp if i not in invited]
            self.t += 1
            
 
        return next_state, self.reward, self.done  #hp: revise 
    
    #the simple state transition process
    def transition(self, invited):#q is probability being present
        present = []
        absent = []
        for i in invited:
            present.append(i) if random.random() <= self.q else absent.append(i)
        #[present.append(i) for i in invited if random.random() <= q]
        return present, absent


    def reset(self):
        self.A = nx.to_numpy_matrix(self.G)
        self.t = 0
        self.done = False
        self.reward = 0
        self.state=np.zeros((2, self.N))
        self.feasible_actions = list(range(self.N))
        nx.set_node_attributes(self.G, 0, 'attr')

if __name__ == '__main__':
    graph_index = 2
    graph_list = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    graph_name = graph_list[graph_index]
    path = 'graph_data/' + graph_name + '.txt'
    G = nx.read_edgelist(path, nodetype=int)
    mapping = dict(zip(G.nodes(),range(len(G))))
    G = nx.relabel_nodes(G,mapping)
    print('selected graph: ', graph_name)
    print('graph size: ', len(G.nodes))
    env=NetworkEnv(G=G)


    rewards = [ ]
    env.reset()
    for i in range(10):
        env.reset()
        while(env.done == False):
            #print('step: ', env.t)
            action = random.sample(env.feasible_actions, env.budget) 
            # print(action)
            env.step(action)
        present = []
        absent = []
        invited = []
        for v in env.G.nodes:
            if env.G.nodes[v]['attr']==1:
                present.append(v)
                invited.append(v)
            if env.G.nodes[v]['attr']==2:
                absent.append(v)
                invited.append(v)
        #print('invited: ', invited)
        #print('present: ', present)
        #print('absent: ', absent)
        #print(env.reward)
        rewards.append(env.reward)
    print('average reward for random policy is: {}, std is: {}'.format(sum(rewards)/10, np.std(rewards)))

    rewards = []
    for i in range(10):
        env.reset()
        actions = []
        presents = []
        while(env.done == False):
            #print('step: ', env.t)
            degree=nx.degree(G)
            degree = [val for (node, val) in G.degree()]
            # print(degree)
            action=[]
            for i in range(env.budget):
                max_degree=0
                for v in env.feasible_actions:
                    if degree[v]>max_degree and v not in action:
                        action_v=v
                        max_degree=degree[v]
                action.append(action_v)
            actions.append(action)
            invited = action
            present, _ = env.transition(action)
            presents.append(present)
            env.step(action)
        rewards.append(env.reward) 
        print('----------------------------------------------')
        print('episode: ', i)
        print('reward: ', env.reward)
        print('invited: ', actions)
        print('present: ', presents)
    print('average reward for maxdegree policy is: {}, std is: {}'.format(sum(rewards)/10, np.std(rewards)))






