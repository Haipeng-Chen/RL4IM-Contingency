from copy import deepcopy
import random
import networkx as nx
import numpy as np

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
                     if random.random() < q:
                         T.append(v)
                     else:
                         V1.append(v)
                 elif v in V1 and v not in V2:
                     if random.random() < (1-(1-p**2)-q)/(1-q): #q+(1-q)x=(1-(1-p^2)), x=(1-(1-p^2)-q)/(1-q)
                         T.append(v)
                     else:
                         V2.append(v) #with prob ()
                 elif v in V2:
                     if random.random() < p:
                         T.append(v)
    return T



def runSC (G, S, d=1 ):
    ''' Runs independent cascade model.
    Input: G -- networkx graph object
    S -- initial list of vertices
    d -- propagation probability
    Output: T -- resulted influenced set of vertices (including S)
    '''
    i = 0
    while i < len(T):
        for v in G[T[i]]: # for neighbors of a selected node
            if v not in T: # if it wasn't selected yet
                w = G[T[i]][v]['weight'] # count the number of edges between two nodes
                if w==1:
                    if random() <= 1 - (1-q)**w: # if at least one of edges propagate influence
                        print T[i], 'influences', v
                        T.append(v) 
                else:
                    if random() <= 1 - (1-p)**w: # if at least one of edges propagate influence
                        print T[i], 'influences', v
                        T.append(v)
        i += 1
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

if __name__ == '__main__':
    g = nx.erdos_renyi_graph(100,0.5)
    #for u,v in g.edges():
        #g[u][v]['p'] = 0.02
    S = random.sample(g.nodes, 10)
    #S = set(S)
    #T = runIC(g, S)
    #print(T)
    infl_mean, infl_std = runIC_repeat(g, S, p=1, sample=1000)
    print(infl_mean, infl_std)





