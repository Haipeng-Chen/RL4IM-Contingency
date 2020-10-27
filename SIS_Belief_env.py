import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor


class EpidemicEnv(object):

    def __init__(self, graph,budget_c=0.1,Initial_I=0.5, infect_prob=0.05, cure_prob=0.05):
        self.graph=graph
        self.n=len(graph)
        self.budget=int(budget_c*self.n)
        self.Initial_I=Initial_I
        self.infect_prob=infect_prob
        self.cure_prob=cure_prob
        self.true_state = np.zeros(self.n)
        self.all_nodes= list(range(self.n))
        self.possible_nodes= list(range(self.n))
        self.observation=[]
        self.A=nx.to_numpy_matrix(graph)
        self.belief_state=self.Initial_I * np.ones(self.n)
        self.T=100
        self.Temperature=1
        self.belief_regressor=ExtraTreesRegressor()
        
    def reset(self):
        self.all_nodes= list(range(self.n))
        self.true_state = np.zeros(self.n)
        self.true_state[random.sample(self.all_nodes, int(self.n*self.Initial_I))] = 1
        # self.belief_state=self.Initial_I * np.ones(self.n)
        self.belief_state=self.Temperature*self.true_state +(1-self.Temperature)*self.Initial_I * np.ones(self.n)
        # self.possible_nodes=[v for v in self.all_nodes if self.true_state[v]==1]
        self.A=nx.to_numpy_matrix(self.graph)
        self.possible_nodes= list(range(self.n))
        self.nature_cure_list   = []
        self.nature_infect_list = []
        self.all_nodes= list(range(self.n))
        self.observation=[]
        self.t=0
        
    def perform(self,active_screen_list):
        S_true=self.true_state.copy()
        S_belief=self.belief_state.copy()
        RL_reward_Initial=sum(self.true_state)
        # print(self.true_state)
        if self.is_action_legal(active_screen_list):
            self.true_state[list(active_screen_list)]=0
        else:
            print("Exceed Budget")
        RL_reward_Initial-=sum(self.true_state)
        Real_reward=self.calc_reward()
        RL_reward=self.Temperature*RL_reward_Initial/(self.budget)+(1-self.Temperature)*Real_reward/(self.n)
        if sum(self.true_state)<(self.budget):
            RL_reward=1
        next_true_state=self.true_state.copy()
        self.nature_cure_list=[v for v in self.all_nodes if self.true_state[v]==1 and random.uniform(0,1)<self.cure_prob]
        self.nature_infect_list=[v for v in self.all_nodes if self.true_state[v]==0 and random.uniform(0,1)<1-(1-self.infect_prob)**(np.inner(self.true_state, self.A[v].A1))]
        next_true_state[self.nature_cure_list]=0
        next_true_state[self.nature_infect_list]=1
        self.observationII=[v for v in self.all_nodes if self.true_state[v]==1 and v not in self.nature_cure_list]
        self.observationIS=self.nature_cure_list
        self.observationSI=self.nature_infect_list
        self.observationSS=[v for v in self.all_nodes if self.true_state[v]==0 and v not in self.nature_infect_list]
        next_belief_state=self.belief_transition(active_screen_list,self.belief_state)
        self.true_state=next_true_state.copy()
        self.belief_state=next_belief_state
        self.t+=1
        self.possible_nodes=[v for v in self.all_nodes if self.belief_state[v]>=self.Temperature*max(self.belief_state)]
        return S_true,S_belief,RL_reward, Real_reward

    def belief_transition(self, previous_action, previous_belief):
        previous_belief[self.observationIS]=1
        belief_state = np.array(previous_belief)
        previous_belief[self.observationII]=self.Temperature*1+(1-self.Temperature)*previous_belief[self.observationII]
        previous_belief[self.observationSI]=(1-self.Temperature)*previous_belief[self.observationSI]
        previous_belief[self.observationSS]=(1-self.Temperature)*previous_belief[self.observationSS]

            
        previous_belief[previous_action]=0
        
        indegree_prob=np.ones(self.n)-self.infect_prob*previous_belief
        Prob=[np.prod([indegree_prob[u] for u in self.A[:,v].nonzero()[0]]) for v in self.all_nodes]
        belief_state=previous_belief+(np.ones(self.n)-previous_belief)*(np.ones(self.n)-Prob) # we know all the reported already thus no (1-c)
        belief_state[self.observationIS]=0
        belief_state[self.observationII]=self.Temperature*1+(1-self.Temperature)*belief_state[self.observationII]
        belief_state[self.observationSI]=self.Temperature*1+(1-self.Temperature)*belief_state[self.observationSI]
        belief_state[self.observationSS]=(1-self.Temperature)*belief_state[self.observationSS]

        return belief_state


    def calc_reward(self):
        reward=self.n-sum(self.true_state)
        return reward    

    def calc_reward2(self):
        reward=-sum(self.true_state)
        return reward  

    def is_action_legal(self,active_screen_list):
        if(len(list(active_screen_list))<=self.budget):
            return True
        else:
            return False    

if __name__ == '__main__':
    print('Here goes nothing')
    Graph_List=['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    Graph_index=5
    Graph_name=Graph_List[Graph_index]
    path='graph_data/'+Graph_name+'.txt'
    G = nx.read_edgelist(path, nodetype=int)
    mapping=dict(zip(G.nodes(),range(len(G))))
    g=nx.relabel_nodes(G,mapping)
    env=EpidemicEnv(graph=g)
    
    env.reset()
    true_I=[]
    belief_I=[]
    Reward=[]
    for i in range(100):
        
        action=random.sample(env.all_nodes,min(len(env.all_nodes),env.budget))
        S1,_,RL_R,R=env.perform(action)
        true_I.append(sum(S1))
        Reward.append(R)
    plt.plot(range(len(true_I)),true_I)
    # plt.plot(range(len(belief_I)),belief_I)
    print(sum(Reward))
    