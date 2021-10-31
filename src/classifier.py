import dgl
from dgl.nn.pytorch import GraphConv
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, dropout):
        super(Classifier, self).__init__()
        self.layers = nn.ModuleList([
            GraphConv(in_dim, hidden_dim),
            GraphConv(hidden_dim, hidden_dim),
        ])
        self.classify = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = F.relu(conv(g,h))
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)
