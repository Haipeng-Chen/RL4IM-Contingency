from copy import deepcopy
import random
import networkx as nx
import numpy as np
from src.agent.baseline import *
from multiprocessing import Process, Manager

import pdb
import time

#RIS
import matplotlib.pyplot as plt
from random import uniform, seed
#import numpy as np
import pandas as pd
#import time
# from igraph import *
#import random
from collections import Counter


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
                Neighbor_fraction[v]/=len(G[v])
                Current_influence[v]=(Neighbor_fraction[v]/(2*d))**2/((Neighbor_fraction[v]/(2*d))**2+(1-Neighbor_fraction[v]/d)**2)
                if Current_influence[v]>Threshold[v]:
                    T.append(v)
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

"""
    REF: https://hautahi.com/im_ris
"""

def get_RRS(G,p):   
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            p:  Disease propagation probability
    Return: A random reverse reachable set expressed as a list of nodes
    """
    
    # Step 1. Select random source node
    source = random.choice(np.unique(G['source']))
    # Step 2. Get an instance of g from G by sampling edges  
    g = G.copy().loc[np.random.uniform(0,1,G.shape[0]) < p]

    # Step 3. Construct reverse reachable set of the random source node
    new_nodes, RRS0 = [source], [source]   
    while new_nodes:
        
        # Limit to edges that flow into the source node
        temp = g.loc[g['target'].isin(new_nodes)]

        # Extract the nodes flowing into the source node
        temp = temp['source'].tolist()

        # Add new set of in-neighbors to the RRS
        RRS = list(set(RRS0 + temp))

        # Find what new nodes were added
        new_nodes = list(set(RRS) - set(RRS0))

        # Reset loop variables
        RRS0 = RRS[:]

    return RRS



def ris(G,k,p=0.5,mc=1000):    
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            k:  Size of seed set
            p:  Disease propagation probability
            mc: Number of RRSs to generate
    Return: A seed set of nodes as an approximate solution to the IM problem
    """
    
    # Step 1. Generate the collection of random RRSs
    start_time = time.time()
    R = [get_RRS(G,p) for _ in range(mc)]

    # Step 2. Choose nodes that appear most often (maximum coverage greedy algorithm)
    SEED, timelapse = [], []
    for _ in range(k):
        
        # Find node that occurs most often in R and add to seed set
        flat_list = [item for sublist in R for item in sublist]
        seed = Counter(flat_list).most_common()[0][0]
        SEED.append(seed)
        
        # Remove RRSs containing last chosen seed 
        R = [rrs for rrs in R if seed not in rrs]
        
        # Record Time
        timelapse.append(time.time() - start_time)
    
    return sorted(SEED), timelapse


def IC_celf(G,S,p=0.5,mc=1000):
    """
    Input:  G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            S:  Set of seed nodes
            p:  Disease propagation probability
            mc: Number of Monte-Carlo simulations
    Output: Average number of nodes influenced by the seed nodes
    """
    
    # Loop over the Monte-Carlo Simulations
    spread = []
    for _ in range(mc):
        
        # Simulate propagation process      
        new_active, A = S[:], S[:]
        while new_active:
            
            # Get edges that flow out of each newly active node
            temp = G.loc[G['source'].isin(new_active)]

            # Extract the out-neighbors of those nodes
            targets = temp['target'].tolist()

            # Determine those neighbors that become infected
            success  = np.random.uniform(0,1,len(targets)) < p
            new_ones = np.extract(success, targets)
            
            # Create a list of nodes that weren't previously activated
            new_active = list(set(new_ones) - set(A))
            
            # Add newly activated nodes to the set of activated nodes
            A += new_active
            
        spread.append(len(A))
        
    return np.mean(spread)


def celf(G,k,p=0.5,mc=1000):   
    """
    Inputs: G:  Ex2 dataframe of directed edges. Columns: ['source','target']
            k:  Size of seed set
            p:  Disease propagation probability
            mc: Number of Monte-Carlo simulations
    Return: A seed set of nodes as an approximate solution to the IM problem
    """
      
    # --------------------
    # Find the first node with greedy algorithm
    # --------------------
    
    # Compute marginal gain for each node
    candidates, start_time = np.unique(G['source']), time.time()
    marg_gain = [IC_celf(G,[node],p=p,mc=mc) for node in candidates]

    # Create the sorted list of nodes and their marginal gain 
    Q = sorted(zip(candidates,marg_gain), key = lambda x: x[1],reverse=True)

    # Select the first node and remove from candidate list
    S, spread, Q = [Q[0][0]], Q[0][1], Q[1:]
    timelapse = [time.time() - start_time]
    
    # --------------------
    # Find the next k-1 nodes using the CELF list-sorting procedure
    # --------------------
    
    for _ in range(k-1):    

        check = False      
        while not check:
            
            # Recalculate spread of top node
            current = Q[0][0]
            
            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current,IC_celf(G,S+[current],p=p,mc=mc) - spread)

            # Re-sort the list
            Q = sorted(Q, key = lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = Q[0][0] == current

        # Select the next node
        S.append(Q[0][0])
        spread = Q[0][1]
        timelapse.append(time.time() - start_time)
        
        # Remove the selected node from the list
        Q = Q[1:]
    
    return sorted(S), timelapse

def make_edge_df(G):
    edges = {}
    for source, target in G.edges():

        if not edges.get('source'):
            edges['source'] = [source]
        else:
            edges['source'].append(source)

        if not edges.get('target'):
            edges['target'] = [target]
        else:
            edges['target'].append(target)
    return pd.DataFrame(edges)

def runIC_estimate(G, S, p=0.01, sample=1000):
    g_d = make_edge_df(G)
    infl_list = []
    R = [get_RRS(g_d,p) for _ in range(sample)]
    flat_list = [item for sublist in R for item in sublist]
    infl=0
    for v in S:
        infl+=Counter(flat_list)[v]
    infl=infl*len(G)/sample
    return infl


if __name__ == '__main__':
    # g = nx.erdos_renyi_graph(200,0.5)
    graph_index=2
    graph_list = ['test_graph','Hospital','India','Exhibition','Flu','irvine','Escorts','Epinions']
    graph_name = graph_list[graph_index]
    path = 'graph_data/' + graph_name + '.txt'
    G = nx.read_edgelist(path, nodetype=int)
    mapping = dict(zip(G.nodes(),range(len(G))))
    g = nx.relabel_nodes(G,mapping)
    #for u,v in g.edges():
        #g[u][v]['p'] = 0.02
    
    budget=10
    S = random.sample(g.nodes, budget)
    def f_multi(x):
        s=list(x)
        val,_=runIC_repeat(G=g, S=s, p=0.05, sample=20)
        return val
    # S, obj=greedy(range(len(g)),budget,f_multi)
    #T = runIC(g, S)
    #print(T)
    start_time = time.time()
    infl_mean, infl_std = runIC_repeat(g, S, p=0.1, sample=1000)
    infl_mean_e = runIC_estimate(g, S, p=0.1, sample=1000)
    # infl_mean, infl_std = runLT_repeat(g, S, l=0.01, sample=1000)
    # infl_mean, infl_std = runSC_repeat(g, S, d=1, sample=1000)
    #infl_mean, infl_std = parallel_influence(g, S, times=10, sample=10)
    print('runtime is: ', time.time()-start_time)
    print('Actual: ',infl_mean, infl_std)
    print('Estimate: ',infl_mean_e)
