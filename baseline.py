# baselines for influence maximization, potentially include maxdegree, greedy and adaptive greedy
import numpy as np

def greedy(items, budget, f):
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
        val, u = heapq.heappop(upper_bounds)
        new_total = f(S.union(set([u])))
        new_val =  new_total - starting_objective
        #lazy evaluation of marginal gains: just check if beats the next highest upper bound up to small epsilon
        if new_val >= -upper_bounds[0][0] - 0.01:
            S.add(u)
            starting_objective = new_total
        else:
            heapq.heappush(upper_bounds, (-new_val, u))
    return S, starting_objective

def multi_to_set(f, n):
    '''
    Takes as input a function defined on indicator vectors of sets, and returns
    a version of the function which directly accepts sets
    '''
    def f_set(S):
        return f(indicator(S, n))
    return f_set

def indicator(S, n):
    x = np.zeros(n)
    x[list(S)] = 1
    return x