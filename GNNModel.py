import torch
import torch_geometric
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,SAGEConv
import numpy as np
import pdb
from torch.nn import Linear


# Defining the GCN model
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GNNModel, self).__init__()
        torch.manual_seed(1234)
        self.conv1 = GCNConv(input_dim, 4)  # Simply define the input features and output features
        self.conv2 = GCNConv(4, 4)
        self.conv3 = GCNConv(4, 2)
        self.classifier = Linear(2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = self.conv1(x, edge_index)  # Input feature and adjacency matrix (note the format, the kind above)

        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()
        h = self.conv3(h, edge_index)
        h = h.tanh()
        # classification layer
        out = self.classifier(h)
        return out, h


