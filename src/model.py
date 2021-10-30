import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Model, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, g):
        # Use node degree as the initial node feature.
        h = g.in_degrees().view(-1, 1).float()
        # Perform graph convolution and activation function.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        g.ndata['h'] = h
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
