import numpy as np
import networkx as nx
import collections


# seed = np.random.seed(120)

class Graph:
    def __init__(self, g, graph_type=None, cur_n=None, p=None, m=None, seed=None):
        if graph_type == 'erdos_renyi':
            self.g = nx.erdos_renyi_graph(n=cur_n, p=p, seed=seed)
        elif graph_type == 'powerlaw':
            self.g = nx.powerlaw_cluster_graph(n=cur_n, m=m, p=p, seed=seed)
        elif graph_type == 'barabasi_albert':
            self.g = nx.barabasi_albert_graph(n=cur_n, m=m, seed=seed)
        elif graph_type =='gnp_random_graph':
            self.g = nx.gnp_random_graph(n=cur_n, p=p, seed=seed)
        else:
            self.g = g

    @classmethod
    def create_graph(cls, g):
        return cls(g)

    def nodes(self):
        return nx.number_of_nodes(self.g)

    def edges(self):
        return self.g.edges()

    def neighbors(self, node):
        return nx.all_neighbors(self.g, node)

    def average_neighbor_degree(self, node):
        return nx.average_neighbor_degree(self.g, nodes=node)

    def adj(self):
        return nx.adjacency_matrix(self.g)

    def __len__(self):
        return len(self.g)
