import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

aggregation_function_generation = 'mean' # either mean or add
aggregation_function = 'mean' # either mean or add

class NaiveGCN(nn.Module):
    def __init__(self, node_feature_size, gcn_hidden_layer_sizes=[8,16,8], nn_hidden_layer_sizes=32):
        super(NaiveGCN, self).__init__()

        r0 =node_feature_size
        r1, r2, r3 = gcn_hidden_layer_sizes
        n1 = nn_hidden_layer_sizes

        # Define the layers of gcn
        self.gcn1 = GraphConv(r0, r1, aggr=aggregation_function)
        self.gcn2 = GraphConv(r1, r2, aggr=aggregation_function)
        self.gcn3 = GraphConv(r2, r3, aggr=aggregation_function)
        # self.gcn4 = GraphConv(r3, r4, aggr=aggregation_function)

        # Define the layers of NN to predict the attractiveness function for every node
        self.fc1 = nn.Linear(r3, n1)
        self.fc2 = nn.Linear(n1, 1)
        # self.fc3 = nn.Linear(n2, 1)

        # self.activation = nn.Softplus()
        # self.activation = F.relu
        self.activation = nn.Sigmoid()

    def forward(self, x, edge_index):
        x = self.activation(self.gcn1(x, edge_index))
        x = self.activation(self.gcn2(x, edge_index))
        x = self.activation(self.gcn3(x, edge_index))
        # x = self.activation(self.gcn4(x, edge_index))

        # x = self.dropout(x)
        x = self.activation(self.fc1(x))
        # x = self.activation(self.fc2(x))
        x = self.fc2(x)

        return x


