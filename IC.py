from copy import deepcopy
import random
import networkx as nx
import numpy as np
from baseline import *
from multiprocessing import Process, Manager

import pdb
import time

def runIC (G, S, p=0.01 ):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial list of vertices
    p -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    
    T = deepcopy(S) # copy already selected nodes
    # ugly C++ version
    #i = 0
    #while i < len(T):
        #for v in G[T[i]]: # for neighbors of a selected node
            #if v not in T: # if it wasn't selected yet
                #w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                #if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                    #print T[i], 'influences', v
                    #T.append(v)
        #i += 1

    # neat pythonic version
    # legitimate version with dynamically changing list: http://stackoverflow.com/a/15725492/2069858
    for u in T: # T may increase size during iterations
         for v in G[u]: # check whether new node v is influenced by chosen node u
             #w = G[u][v]['weight']
             w = 1
             if v not in T and random.random() < 1 - (1-p)**w:
                 T.append(v)
    return T

def runDIC (G, S, p=0.01, q=0.001 ):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial list of vertices
    V1 -- set of first attpemt influenced node
    V2 -- set of second attpemt influenced node
    p -- propagation probability
    q -- 1 neighbor propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    T = deepcopy(S)
    V1=[]
    V2=[]
    i = 0
    for u in T: # T may increase size during iterations
         for v in G[u]: # check whether new node v is influenced by chosen node u
             #w = G[u][v]['weight']
             w = 1
             if v not in T:
                 if v not in V1:
                     if random.random() < q: # First infect as small prob
                         T.append(v)
                     else:
                         V1.append(v)
                 elif v in V1 and v not in V2:
                     if random.random() < (1-(1-p**2)-q)/(1-q): #q+(1-q)x=(1-(1-p^2)), x=(1-(1-p^2)-q)/(1-q)
                         T.append(v)
                     else:
                         V2.append(v) #the node is atempted by more then 2 times
                 elif v in V2:
                     if random.random() < p:
                         T.append(v)
    return T

def runLT (G, S, l=0.01 ):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial list of vertices
    l -- weight #revise?s
    Output: T -- resulted influenced set of vertices (including S)
    '''
    
    T = deepcopy(S)
    Threshold=np.random.rand(len(G)) #uniform random threshold
    Any_Change=True    
    while(Any_Change): #loop until no newly infected
        Any_Change=False
        Current_influence=np.zeros(len(G)) #Reset influence
        for v in T:
            Current_influence[v]=1 #Set influenced nodes above threshold
        for u in T: 
            for v in G[u]:
                if v not in T: 
                    Current_influence[v]+=l #Add influence to neighbor
                    if Current_influence[v]>Threshold[v]:#If some nodes are newly infected
                        T.append(v)
                        Any_Change=True#Set change to true            
    return T

def runSC (G, S, d=1 ):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial list of vertices
    d -- fraction coefficient
    Output: T -- resulted influenced set of vertices (including S)
    '''
    Threshold=np.random.rand(len(G))
    T = deepcopy(S)
    Any_Change=True    
    while(Any_Change):
        Any_Change=False
        Current_influence=np.zeros(len(G))
        Neighbor_fraction=np.zeros(len(G)) #fraction of neighbor got infected
        
        for v in range(len(G)):
            if v not in T:
                for u in G[v]:
                    if u in T:
                        Neighbor_fraction[v]+=1
        for v in range(len(G)):
            Neighbor_fraction[v]/=len(G[v])
        # Count fraction of neighbor, a bit ugly
        for v in range(len(G)):
            Current_influence[v]=(Neighbor_fraction[v]/(2*d))**2/((Neighbor_fraction[v]/(2*d))**2+(1-Neighbor_fraction[v]/d)**2)
        for v in T:
            Current_influence[v]=1
        # Calculate current accumulated thrshold, bellow same as LT
        NewT=[]
        for v in range(len(G)):
            if Current_influence[v]>Threshold[v]:
                NewT.append(v)
        if len(NewT)>len(T):
            T = deepcopy(NewT)
            Any_Change=True
    return T


def runIC_repeat(G, S, p=0.01, sample=1000):
    infl_list = []
    for i in range(sample):
        T = runIC(G, S, p=p)
        influence = len(T)
        infl_list.append(influence)
    infl_mean = np.mean(infl_list)
    infl_std = np.std(infl_list)

    return infl_mean, infl_std 

def runDIC_repeat(G, S, p=0.01, q=0.001, sample=1000):
    infl_list = []
    for i in range(sample):
        #if i%100==0:
            #print('i in runIC_repeat: ', i)
        T = runDIC(G, S, p=p, q=q)
        influence = len(T)
        infl_list.append(influence)
    infl_mean = np.mean(infl_list)
    infl_std = np.std(infl_list)

    return infl_mean, infl_std

def runLT_repeat(G, S, l=0.01, sample=1000):
    infl_list = []
    for i in range(sample):
        T = runLT(G, S,l=l)
        influence = len(T)
        infl_list.append(influence)
    infl_mean = np.mean(infl_list)
    infl_std = np.std(infl_list)

    return infl_mean, infl_std 

def runSC_repeat(G, S, d=1, sample=1000):
    infl_list = []
    for i in range(sample):
        T = runSC(G, S,d=d)
        influence = len(T)
        infl_list.append(influence)
    infl_mean = np.mean(infl_list)
    infl_std = np.std(infl_list)

    return infl_mean, infl_std 

def influence_wrapper(l,G,S,sample):
    ans = runIC_repeat(G, S, p=0.01, sample=sample)
    l.append(ans[0])

def parallel_influence(G, S, times=10, sample=1000,PROCESSORS=4):
    

    
    l = Manager().list()
    processes = [Process(target=influence_wrapper, args=(l, G, S,sample)) for _ in range(times)]
    i=0
    while i<len(processes):
        j = i+PROCESSORS if i+PROCESSORS < len(processes) else len(processes)-1
        ps = processes[i:j]
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        i+= PROCESSORS
    l = list(l)
    return np.mean(l) ,np.std(l)

if __name__ == '__main__':
    g = nx.erdos_renyi_graph(100,0.5)
    #for u,v in g.edges():
        #g[u][v]['p'] = 0.02
    budget=40
    S = random.sample(g.nodes, budget)
    def f_multi(x):
        s=list(x)
        val,_=runIC_repeat(G=g, S=s, p=0.01, sample=1000)
        return val
    # S, obj=greedy(range(len(g)),budget,f_multi)
    #T = runIC(g, S)
    #print(T)
    start_time = time.time()
    #infl_mean, infl_std = runIC_repeat(g, S, p=0.1, sample=1000)
    #infl_mean, infl_std = runLT_repeat(g, S, l=0.01, sample=1000)
    infl_mean, infl_std = runSC_repeat(g, S, d=1, sample=1000)
    #infl_mean, infl_std = parallel_influence(g, S, times=10, sample=10)
    print('runtime is: ', time.time()-start_time)
    print(infl_mean, infl_std)
