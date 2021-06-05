import copy

import numpy as np
import networkx as nx
import collections


class Graph:
    def __init__(self, g=None, graph_type=None, cur_n=None, p=None, m=None, seed=None, args=None, is_train=True):
        self.seed = seed
        self.args = args

        if seed is not None:
            np.random.seed(seed)

        if g is not None:  # load customized graphs
            self.g = g
            self.cur_n = nx.number_of_nodes(self.g)
            self.max_node_num = self.cur_n

            self.orig_g = copy.deepcopy(self.g)
            
            if self.args.sample_nodes_ratio < 1.0: 
                self.size = int(np.floor(self.args.sample_nodes_ratio * len(self.orig_g.nodes)))
                self.init_sub_graph()
        
            return 
        
        self.max_node_num = cur_n + (self.args.graph_node_var if self.args.model_scheme != 'normal' else 0)
        
        if args.model_scheme != 'normal' or not is_train:
            cur_n += np.random.choice(range(-self.args.graph_node_var, self.args.graph_node_var+1, 1))

        self.cur_n = cur_n

        # create graph with networkx
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
        
        self.orig_g = copy.deepcopy(self.g)

    @classmethod
    def create_graph(cls, g):
        return cls(g=g)

    @property
    def node(self):
        return nx.number_of_nodes(self.g)

    @property
    def nodes(self):
        return self.g.nodes()

    @property
    def edges(self):
        return self.g.edges()

    @property
    def neighbors(self, node):
        return nx.all_neighbors(self.g, node)

    def average_neighbor_degree(self, node):
        return nx.average_neighbor_degree(self.g, nodes=node)

    @property
    def adj(self):
        return nx.adjacency_matrix(self.g)

    def __len__(self):
        return len(self.g)

    def init_sub_graph(self):
        sampled_nodes = np.random.choice(list(self.orig_g.nodes()), size=self.size, replace=False)
        _temp_g = self.orig_g.subgraph(sampled_nodes.tolist()).copy()

        edges = list(_temp_g.edges())

        nodes = sampled_nodes.tolist()
        index_map = {node: idx for idx, node in zip(range(len(sampled_nodes)), nodes)}
        _temp_g = nx.relabel_nodes(_temp_g, index_map)

        self.g = _temp_g
        self.cur_n = nx.number_of_nodes(self.g)
        self.max_node_num = self.cur_n

        #print(f'[INFO] sampled_nodes len: {len(sampled_nodes)}, sampled_graph #nodes: {len(_temp_g.nodes)}, self.g #nodes: {len(self.g.nodes)}')
