# baselines for influence maximization, potentially include maxdegree, greedy and adaptive greedy
import pdb 
import time

import operator
import numpy as np
import networkx as nx



def lazy_greedy(items, budget, f):
    '''
    Generic greedy algorithm to select budget number of items to maximize f.
    
    Employs lazy evaluation of marginal gains, which is only correct when f is submodular.
    '''
    import heapq
    if budget >= len(items):
        S = set(items)
        return S, f(S)
    upper_bounds = [(-f(set([u])), u) for u in items]    
    heapq.heapify(upper_bounds)
    starting_objective = f(set())
    S  = set()
    #greedy selection of K nodes
    while len(S) < budget:
        #print(len(S))
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound up to small epsilon
        if new_val >= -upper_bounds[0][0] - 0.01:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return list(S), starting_objective


def greedy(items, budget, f):
    S = set()
    for i in range(budget):
        #print(i)
        Inf = dict() # influence for nodes not in S
        for v in items:
            if v not in S:
                SS=S.copy()
                Inf[v] = f(SS.union(set([v])))
        # print(max(Inf.items(), key=operator.itemgetter(1)))
        u, val = max(Inf.items(), key=operator.itemgetter(1))
        starting_objective =val
        S.add(u)
    return list(S), starting_objective


def lazy_adaptive_greedy(items, budget, f, S_prev=[]):
    '''
    Generic greedy algorithm to select budget number of items to maximize f.
    
    Employs lazy evaluation of marginal gains, which is only correct when f is submodular.
    '''
    import heapq
    if budget >= len(items):
        S = set(items)
        return S, f(S)
    upper_bounds = [(-f(set([u])), u) for u in items]    
    heapq.heapify(upper_bounds)
    starting_objective = f(set())
    action=set()
    S  = set(S_prev)
    #greedy selection of K nodes
    start = time.time()
    while len(action) < budget:
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound up to small epsilon
        if new_val >= -upper_bounds[0][0] - 0.01:
            S.add(u)
            action.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    #print('runtime for finding one greedy action set is: ',time.time()-start)
    #pdb.set_trace()
    return list(action), starting_objective


def adaptive_greedy(items, budget, f, S_prev=[]):
    action=set()
    S = set(S_prev)
    for i in range(budget):
        Inf = dict() # influence for nodes not in S
        for v in items:
            if v not in S:
                SS=S.copy()
                Inf[v] = f(SS.union(set([v])))
        # print(max(Inf.items(), key=operator.itemgetter(1)))
        u, val = max(Inf.items(), key=operator.itemgetter(1))
        starting_objective =val
        S.add(u)
        action.add(u)
    return list(action), starting_objective


def max_degree(feasible_actions, G, budget):
    degree=nx.degree(G)
    degree = [val for (node, val) in G.degree()]
    # print(degree)
    action=[]
    for i in range(budget):
        max_degree=0
        for v in feasible_actions:
            if degree[v]>max_degree and v not in action:
                action_v=v
                max_degree=degree[v]
        action.append(action_v)
    return action