# another graph sage model
# based on random walks


import torch
from torch.nn import Linear, LogSoftmax, Sigmoid
from torch_geometric.nn import SAGEConv



class GraphSAGE(torch.nn.Module):
    
    def __init__(self, node_attr_dim, hidden_dim=32, node_out_dim=20, seed=None, hop=2, aggr='mean'): 
        super(GraphSAGE, self).__init__()

        assert hop >= 2
        self.hop = hop
        assert aggr in ['add', 'mean', 'max']
        self.aggr = aggr

        self.node_attr_dim = node_attr_dim
        self.hidden_dim = hidden_dim
        self.node_out_dim = node_out_dim
        self.seed = seed

        if self.seed is not None: torch.manual_seed(self.seed)
        
        # we don't want to use one's identity to predict one's identity
        self.conv1 = SAGEConv(self.node_attr_dim, self.hidden_dim, normalize=True, aggr=aggr)
        self.conv2 = SAGEConv(self.hidden_dim, self.hidden_dim, normalize=True, aggr=aggr)
        self.convn = []
        for __ in range(self.hop - 2): self.convn.append(SAGEConv(self.hidden_dim, self.hidden_dim, normalize=True, aggr=aggr))

        self.classifier = Linear(self.hidden_dim, self.node_out_dim)
        self.logsoftmax = LogSoftmax(dim=1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x, edge_index): 
        
        h = torch.tanh(self.conv1(x, edge_index))
        h = torch.tanh(self.conv2(h, edge_index))
        for i in range(self.hop - 2): h = torch.tanh(self.convn[i](h, edge_index))
        
        if self.node_out_dim == 20: 
            out = self.logsoftmax(self.classifier(h))
        elif self.node_out_dim == 1: 
            out = self.sigmoid(self.classifier(h)).view(-1)
        else: raise NotImplementedError()
        return out, h