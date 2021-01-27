import numpy as np
import networkx as nx
import math
import random
import argparse
import pulp
#from influence import influence, parallel_influence
import os
from src.IC import runIC_repeat
from src.IC import runDIC_repeat
from src.IC import runLT_repeat
from src.IC import runSC_repeat
from src.agent.baseline import *

#from IC import runIC_repeat
#from IC import runDIC_repeat
#from IC import runLT_repeat
#from IC import runSC_repeat 
#from baseline import *

import time


class NetworkEnv(object):
    '''
    Environment for peer leader selection process of influence maximization
    
    we consider fully known and static graph first, will potentially extend to: 1) dynamic graph 2) unknown graph 3) influence at each step 4) ...
    G is a nx graph
    node 'attr': a trinary value where 0 is not-selected; 1 is selected and present; 2 is selected but not present; I am planning to put the state and action embedding outside environment 
    state is 3xN binary array, 
    -- 1st row: invited in previous main step and came (=1), 
    -- 2nd row: invited but not come (=1); 
    -- 3rd row: invited in previous sub step (=1) or not (=0) --- it is only useful in states in the sub steps, not updated in env  
    -- elements (=0) on both 1st and 2nd rows are not invited and thus are feasible actions
    note that the 3rd row of state is only updated outside environment (in rl4im.py: greedy_action_GCN() and memory store step)
    '''
    
    def __init__(self, G, T=20, budget=5, propagate_p = 0.1, l=0.05, d=1, q=1, cascade='IC', num_simul=250, graphs=None):
        self.G = G
        self.graphs = graphs
        self.N = len(self.G)
        #self.budget = math.floor(self.N * budget_ratio/T)
        self.budget = budget
        self.A = nx.to_numpy_matrix(self.G)  
        self.propagate_p = propagate_p
        self.l = l
        self.d = d
        self.q = q
        self.T = T
        self.cascade = cascade
        self.num_simul = num_simul
        self.t = 0
        self.done = False
        self.reward = 0
        #self.feasible_actions = list(range(self.N))
        self.state = np.zeros((3, self.N)) 
        self.observation = self.state
        nx.set_node_attributes(self.G, 0, 'attr')

    def step(self, i, pri_action, sec_action):
        #pri_action is a list, sec_action is an int
        #compute reward as marginal contribution of a node
        print('time step: ', i)
        if i == 1:
            seeds = [sec_action]
            #print('seeds are ', seeds)
            self.reward = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
            #print('reward:', self.reward)
            #pdb.set_trace()
        else:
            seeds = []
            [seeds.append(v) for v in range(self.N) if (self.state[0][v]==1 or self.state[2][v]==1)] #I am treating state[2][v]==1 as q=1 TODO: change it to probabilistic
            #print('seeds without {} are {}'.format(sec_action, seeds))
            influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
            #print('influence without is ', influece_without)
            seeds.append(sec_action)
            #print('seeds with it  are ',seeds)
            influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
            self.reward = influence_with - influece_without
            #pdb.set_trace()

        #update feasible actions
        #print('feasible actions:',  self.feasible_actions)
        #self.feasible_actions.remove(sec_action)
        #print('feasible actions:',  self.feasible_actions)

        #update next_state and done      
        if i%self.budget == 0:
        #a primary step
            invited = pri_action
            present, absent = self.transition(invited)
            state=self.state.copy()
            for v in present:
                self.G.nodes[v]['attr']=1 #TODO: remove this?
                self.state[0][v]=1
            for v in absent:
                self.G.nodes[v]['attr']=2
                self.state[1][v]=1
            self.state[2].fill(0)
            if i == self.T:
                #seeds = []
                #[seeds.append(v) for v in range(self.N) if self.G.nodes[v]['attr'] == 1]
                #[seeds.append(v) for v in range(self.N) if self.state[0][v] == 1]
                #self.reward = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
                next_state = None
                self.done = True
            else:
                #self.reward = 0 
                next_state = self.state.copy()
                self.done = False
        else:
        #a secondary step
            self.state[2][sec_action]=1
            next_state = self.state.copy()
            self.done = False

        if i == self.T:  # i%self.budget == 0 may not True when i == self.T
            next_state = None
            self.done = True

        return next_state, self.reward, self.done
            
    
   # def step(self, i, pri_action, sec_action):
   #     #compute reward as marginal contribution of a node
   #     seeds = []
   #     [seeds.append(v) for v in range(self.N) if (self.state[0][v]==1 or self.state[2][v]==1)] #I am treating state[2][v]==1 as q=1 TODO: change it to probabilistic
   #     influece_without = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
   #     seeds.append(sec_action)
   #     influence_with = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
   #     self.reward = influence_with - influece_without

   #     #update feasible actions
   #     self.feasible_actions.remove(sec_action)

   #     #update next_state and done      
   #     if i == self.budget-1:
   #     #a primary step
   #         invited = pri_action
   #         present, absent = self.transition(invited)
   #         state=self.state.copy()
   #         for v in present:
   #             self.G.nodes[v]['attr']=1 #TODO: remove this?
   #             self.state[0][v]=1
   #         for v in absent:
   #             self.G.nodes[v]['attr']=2
   #             self.state[1][v]=1
   #         #numpy.fill(self.state[2])
   #         #TODO: assigne 0 values to all state[2]
   #         if self.t == self.T-1:
   #             #seeds = []
   #             #[seeds.append(v) for v in range(self.N) if self.G.nodes[v]['attr'] == 1]
   #             #[seeds.append(v) for v in range(self.N) if self.state[0][v] == 1]
   #             #self.reward = self.run_cascade(seeds=seeds, cascade=self.cascade, sample=self.num_simul)
   #             next_state = None
   #             self.done = True
   #         else:
   #             #self.reward = 0 
   #             next_state = self.state.copy()
   #             self.done = False
   #             #self.feasible_actions.remove(sec_action)
   #             #feasible_actions_cp = self.feasible_actions.copy()
   #             #self.feasible_actions = [i for i in feasible_actions_cp if i not in invited]
   #             self.t += 1
   #     else:
   #     #a secondary step
   #         self.state[2][sec_action]=1
   #         next_state = self.state.copy()
   #         self.done = False
   #         #self.feasible_actions.remove(sec_action) ########correct? 
   #         
   #     return next_state, self.reward, self.done  
    
    def run_cascade(self, seeds, cascade='IC', sample=1000):
        #print('running cascade')
        #there may be better ways of passing the arguments
        if cascade == 'IC':
            reward, _ = runIC_repeat(self.G, seeds, p=self.propagate_p, sample=sample)
        elif cascade == 'DIC':
            reward, _ = runDIC_repeat(self.G, seeds, p=self.propagate_p, q=0.001, sample=sample)
        elif cascade == 'LT':
            reward, _ = runLT_repeat(self.G, seeds, l=self.l, sample=sample)
        elif cascade == 'SC':
            reward, _ = runSC_repeat(self.G, seeds, d=self.d, sample=sample)
        else:
            assert(False)
        return reward
 
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
        self.state = np.zeros((3, self.N)) ########
        self.observation = self.state
        #self.feasible_actions = list(range(self.N))
        nx.set_node_attributes(self.G, 0, 'attr')

class Environment(NetworkEnv):
    def __init__(self, G, T=1, budget=20, propagate_p = 0.1, l=0.05, d=1, q=1, cascade='IC', num_simul=1000, graphs=None, name='MVC'):
        super().__init__(G=G,
                         T=T,
                         budget=budget,
                         propagate_p=propagate_p,
                         l=l,
                         d=d,
                         q=q,
                         cascade=cascade,
                         num_simul=num_simul,
                         graphs=graphs)
        self.name = name
        self.graph_init = G

    def get_approx(self):
        if self.name == "MVC":
            cover_edge=[]
            edges= list(self.graph_init.edges())
            while len(edges) > 0:
                edge = edges[np.random.choice(len(edges))]
                cover_edge.append(edge[0])
                cover_edge.append(edge[1])
                to_remove=[]
                for edge_ in edges:
                    if edge_[0]==edge[0] or edge_[0]==edge[1]:
                        to_remove.append(edge_)
                    else:
                        if edge_[1]==edge[1] or edge_[1]==edge[0]:
                            to_remove.append(edge_)
                for i in to_remove:
                    edges.remove(i)
            return len(cover_edge)

        elif self.name=="MAXCUT":
            return 1
        else:
            return 'you pass a wrong environment name'


    def get_optimal_sol(self):
        if self.name =="MVC":
            x = list(range(self.graph_init.g.number_of_nodes()))
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMinimize)
            mdl += sum(xv[k] for k in xv)
            for edge in self.graph_init.edges():
                mdl += xv[edge[0]] + xv[edge[1]] >= 1, "constraint :" + str(edge)
            mdl.solve()

            #print("Status:", pulp.LpStatus[mdl.status])
            optimal=0
            for x in xv:
                optimal += xv[x].value()
                #print(xv[x].value())
            return optimal

        elif self.name=="MAXCUT":
            x = list(range(self.graph_init.g.number_of_nodes()))
            e = list(self.graph_init.edges())
            xv = pulp.LpVariable.dicts('is_opti', x,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)
            ev = pulp.LpVariable.dicts('ev', e,
                                       lowBound=0,
                                       upBound=1,
                                       cat=pulp.LpInteger)

            mdl = pulp.LpProblem("MVC", pulp.LpMaximize)

            mdl += sum(ev[k] for k in ev)

            for i in e:
                mdl+= ev[i] <= xv[i[0]]+xv[i[1]]

            for i in e:
                mdl+= ev[i]<= 2 -(xv[i[0]]+xv[i[1]])

            #pulp.LpSolverDefault.msg = 1
            mdl.solve()
            # print("Status:", pulp.LpStatus[mdl.status])
            return mdl.objective.value()


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments of influence maximzation')
    parser.add_argument('--baseline',dest='baseline', type=str, default='ada_greedy',
                help='baseline, could be ada_greedy, random, maxdegree')
    parser.add_argument('--graph_index',dest='graph_index', type=int, default=2,
                help='graph index')
    parser.add_argument('--T', dest='T', type=int, default=1,
                help='time horizon')
    #parser.add_argument('--budget_ratio', dest='budget_ratio', type=float, default=0.06,
                #help='budget ratio; do the math: budget at each step = graph_size*budget_ratio/T')
    parser.add_argument('--budget', dest='budget', type=int, default=20,
                help='budget at each main step')

    parser.add_argument('--cascade',dest='cascade', type=str, default='IC',
                help='cascade model')
    parser.add_argument('--propagate_p', dest='propagate_p', type=float, default=0.1,
                help='influence propagation probability')
    parser.add_argument('--l', dest='l', type=float, default=0.05,
                help='influence of each neighbor in LT cascade')
    parser.add_argument('--d', dest='d', type=float, default=1,
                help='d in SC cascade')
    parser.add_argument('--q', dest='q', type=float, default=1,
                help='probability of invited node being present')
    parser.add_argument('--num_simul',dest='num_simul', type=int, default=1000,
                help='number of simulations for env.step')
    parser.add_argument('--greedy_sample_size',dest='greedy_sample_size', type=int, default=500,
                help='sample size for value estimation of greedy algorithms')

    return parser.parse_args()

# THE FOLLOWING CODE ARE FOR TESTING
if __name__ == '__main__':

    args = arg_parse()
    graph_index = args.graph_index 
    baseline = args.baseline
    T = args.T
    #budget_ratio = args.budget_ratio
    budget = args.budget
    cascade = args.cascade
    propagate_p = args.propagate_p
    l = args.l
    d = args.d
    q = args.q
    num_simul = args.num_simul
    greedy_sample_size = args.greedy_sample_size

    graph_list = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    graph_name = graph_list[graph_index]
    path = 'graph_data/' + graph_name + '.txt'
    G = nx.read_edgelist(path, nodetype=int)
    mapping = dict(zip(G.nodes(),range(len(G))))
    G = nx.relabel_nodes(G,mapping)
    print('selected graph: ', graph_name)
    print('#nodes: ', len(G.nodes))
    print('#edges: ', len(G.edges))
    env=NetworkEnv(G=G, T=T, budget=budget, propagate_p = propagate_p, l=l, d=d, q=q, cascade=cascade)


    rewards = []
    def f_multi(x):
        s=list(x) 
        #print('cascade model is: ', env.cascade)
        val = env.run_cascade(seeds=s, cascade=env.cascade, sample=greedy_sample_size)
        return val

    episodes = 5 
    runtime1 = 0
    runtime2 = 0
    for i in range(episodes):
        print('----------------------------------------------')
        print('episode: ', i)
        env.reset()
        actions = []
        presents = []
        while(env.done == False):
            start = time.time()
            if baseline == 'random':
                action = random.sample(env.feasible_actions, env.budget) 
            elif baseline == 'maxdegree':
                action = max_degree(env.feasible_actions, env.G, env.budget)
            elif baseline == 'ada_greedy':
                action, _ = adaptive_greedy(env.feasible_actions,env.budget,f_multi,presents)
            elif baseline == 'lazy_ada_greedy':
                action, _ = lazy_adaptive_greedy(env.feasible_actions,env.budget,f_multi,presents)
            else:
                assert(False)
            runtime1 += time.time()-start
            start = time.time()
            actions.append(action)
            invited = action
            present, _ = env.transition(action)
            presents+=present
            runtime2 += time.time()-start
            env.step(action)
        rewards.append(env.reward) 
        print('reward: ', env.reward)
        print('invited: ', actions)
        print('present: ', presents)
    print()
    print('----------------------------------------------')
    print('average reward for {} policy is: {}, std is: {}'.format(baseline, np.mean(rewards), np.std(rewards)))
    print('total runtime for action selection is: {}, total runtime for env.step is: {}'.format(runtime1, runtime2))
