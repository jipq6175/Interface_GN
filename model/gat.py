# this is another gat model base


import torch
from torch.nn import Linear, LogSoftmax, Sigmoid
from torch_geometric.nn import GATv2Conv


class GAT(torch.nn.Module):
    
    def __init__(self, node_attr_dim, edge_attr_dim, hidden_dim=32, node_out_dim=20, seed=None, \
                 hop=2, heads=64, dropout=0.2): 
        super(GAT, self).__init__()

        assert hop >= 2
        self.hop = hop

        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.node_out_dim = node_out_dim
        self.heads = heads
        self.dropout = dropout
        self.seed = seed

        if self.seed is not None: torch.manual_seed(self.seed)
        
        # we don't want to use one's identity to predict one's identity
        self.conv1 = GATv2Conv(self.node_attr_dim, self.hidden_dim, add_self_loops=False, heads=self.heads, dropout=self.dropout, edge_dim=self.edge_attr_dim, concat=False)
        self.conv2 = GATv2Conv(self.hidden_dim, self.hidden_dim, add_self_loops=False, heads=self.heads, dropout=self.dropout, edge_dim=self.edge_attr_dim, concat=False)
        self.convn = []
        for __ in range(self.hop - 2): self.convn.append(GATv2Conv(self.hidden_dim, self.hidden_dim, add_self_loops=False, heads=self.heads, dropout=self.dropout, edge_dim=self.edge_attr_dim))

        self.classifier = Linear(self.hidden_dim * self.heads, self.node_out_dim)
        self.logsoftmax = LogSoftmax(dim=1)
        self.sigmoid = Sigmoid()

        
    def forward(self, x, edge_index, edge_attr): 
        h, (__, edge_attention) = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        h = torch.tanh(h)
        h = torch.tanh(self.conv2(h, edge_index, edge_attr=edge_attr, return_attention_weights=False)[0])
        for i in range(self.hop - 2): h = torch.tanh(self.convn[i](h, edge_index, edge_attr=edge_attr, return_attention_weights=True)[0])
        
        if self.node_out_dim == 20: 
            out = self.logsoftmax(self.classifier(h))
        elif self.node_out_dim == 1: 
            out = self.sigmoid(self.classifier(h)).view(-1)
        else: raise NotImplementedError()
        return out, h