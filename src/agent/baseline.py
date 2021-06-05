# CHANGE baselines for influence maximization, adapted from Wilder, Bryan, et al. "End-to-End Influence Maximization in the Field." AAMAS. Vol. 18. 2018.
import time
import networkx as nx


class lazy_adaptive_greedyAgent:
    #Generic greedy algorithm to select budget number of items to maximize f.
    #Employs lazy evaluation of marginal gains, which is only correct when f is submodular.
    def __init__(self):
        self.method = 'lazy_adaptive_greedy'

    def act(self, items, budget, f, S_prev=[]):
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
        return list(action), starting_objective

