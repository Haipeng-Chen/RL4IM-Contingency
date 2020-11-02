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

    node 'attr': a trinary value where 0 is not-selected; 1 is selected and present; 2 is selected but not present; I am planning to put the state and action embedding outside environment 

    state consists of node 'attr' and A
    '''
    
    def __init__(self, G, T=3, budget_ratio=0.2, propagate_p = 0.1):
        self.G = G
        self.N = len(self.G)
        self.budget = math.floor(self.N * budget_ratio/T)
        #self.node_attr =  np.zeros(self.N) #I am planning to put the state and action embedding outside environment 
        self.A = nx.to_numpy_matrix(self.G)  
        self.propagate_p = propagate_p
        self.T = T
        self.t = 0
        self.done = False
        self.reward = 0
        self.feasible_actions = list(range(self.N))
        self.state=np.zeros(self.N)#0: not invited, 1: invited and came. 2: invited and not came
        nx.set_node_attributes(self.G, 0, 'attr')

    def step(self, action):
        #when final setp
        invited = action
        present, absent = self.transition(invited, q=0.6)
        state=self.state.copy()
        for v in present:
            self.G.nodes[v]['attr']=1
            self.state[v]=1
        for v in absent:
            self.G.nodes[v]['attr']=2
            self.state[v]=2
        next_state=self.state.copy()
        if self.t == self.T:
            seeds = []
            [seeds.append(v) for v in range(self.N) if self.G.nodes[v]['attr'] == 1]
            self.reward, _ = runIC_repeat(self.G, seeds, p=self.propagate_p, sample=1000)
            self.done = True
        else:
            #invited = action
            #present, absent = transition(invited, q=0.6)
            #for i in present:
                #self.G.nodes[v]['attr']=1
            #for i in absent:
                #self.G.nodes[v]['attr']=2
            self.reward = 0 #TODO: add an auxilliary reward to warm-start 
            #Han Ching: One idea is to simulate the IM here with given seend.
            feasible_actions_cp = self.feasible_actions.copy()
            self.feasible_actions = [i for i in feasible_actions_cp if i not in invited]
            self.t += 1
            
 
        return state, action, self.reward, next_state #hp: revise 
    
    #the simple state transition process
    def transition(self, invited, q=0.6):#q is probability being present
        present = []
        absent = []
        for i in invited:
            present.append(i) if random.random() <= q else absent.append(i)
        #[present.append(i) for i in invited if random.random() <= q]
        return present, absent


    def reset(self):
        self.A = nx.to_numpy_matrix(self.G)
        self.t = 0
        self.done = False
        self.reward = 0
        self.feasible_actions = list(range(self.N))
        nx.set_node_attributes(self.G, 0, 'attr')

if __name__ == '__main__':
    Graph_List=['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    Graph_index = 1
    Graph_name=Graph_List[Graph_index]
    path='graph_data/'+Graph_name+'.txt'
    G = nx.read_edgelist(path, nodetype=int)
    mapping=dict(zip(G.nodes(),range(len(G))))
    G = nx.relabel_nodes(G,mapping)
    env=NetworkEnv(G=G)


    rewards = [ ]
    env.reset()
    while(env.done == False):
        print('step: ', env.t)
        action = random.sample(env.feasible_actions, env.budget) 
        print(action)
        env.step(action)
        print(env.reward)
        #rewards.append(env.reward)
    #print(sum(rewards))

    '''
    #sanity check for trnasition()
    invited = [1, 2, 3, 4]
    num_invited = 0
    for i in range(1000):
        present,absent = env.transition(invited, q=0.6)
        num_invited += len(present)
    print(num_invited/1000)    
    '''





