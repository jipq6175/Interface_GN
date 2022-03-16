

import torch
from torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d, LogSoftmax, Sigmoid
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch_geometric.nn import MetaLayer



class GraphNet(torch.nn.Module):
    
    def __init__(self, node_attr_dim, edge_attr_dim, graph_attr_dim, \
                 hidden_dim=32, node_output_dim=32, edge_output_dim=32, graph_output_dim=32, node_out_dim=20,\
                 dropout=0.2, aggregation='mean', seed=None, hop=2):
        
        super(GraphNet, self).__init__()
        assert hop >= 2
        if seed is not None: torch.manual_seed(seed)
        self.hop = hop

        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.graph_attr_dim = graph_attr_dim
        
        self.hidden_dim = hidden_dim
        self.node_output_dim = node_output_dim
        self.edge_output_dim = edge_output_dim
        self.graph_output_dim = graph_output_dim

        self.node_out_dim = node_out_dim
        
        self.dropout = dropout
        self.aggregation = aggregation
        
        self.GN1 = MetaLayer(EdgeBlock(self.node_attr_dim, self.edge_attr_dim, self.graph_attr_dim, hidden_dim=self.hidden_dim, output_dim=self.edge_output_dim, dropout=self.dropout), 
                             NodeBlock(self.node_attr_dim, self.edge_output_dim, self.graph_attr_dim, hidden_dim=self.hidden_dim, output_dim=self.node_output_dim, dropout=self.dropout, aggregation=self.aggregation), 
                             GlobalBlock(self.node_output_dim, self.graph_attr_dim, hidden_dim=self.hidden_dim, output_dim=self.graph_output_dim, dropout=self.dropout, node_aggregation=self.aggregation))
        
        self.GN2 = MetaLayer(EdgeBlock(self.node_output_dim, self.edge_output_dim, self.graph_output_dim, hidden_dim=self.hidden_dim, output_dim=self.edge_output_dim, dropout=self.dropout), 
                             NodeBlock(self.node_output_dim, self.edge_output_dim, self.graph_output_dim, hidden_dim=self.hidden_dim, output_dim=self.node_output_dim, dropout=self.dropout, aggregation=self.aggregation), 
                             GlobalBlock(self.node_output_dim, self.graph_output_dim, hidden_dim=self.hidden_dim, output_dim=self.graph_output_dim, dropout=self.dropout, node_aggregation=self.aggregation))
        
        self.GNn = []
        for __ in range(self.hop - 2): 
            self.GNn.append(MetaLayer(EdgeBlock(self.node_output_dim, self.edge_output_dim, self.graph_output_dim, hidden_dim=self.hidden_dim, output_dim=self.edge_output_dim, dropout=self.dropout), 
                                      NodeBlock(self.node_output_dim, self.edge_output_dim, self.graph_output_dim, hidden_dim=self.hidden_dim, output_dim=self.node_output_dim, dropout=self.dropout, aggregation=self.aggregation), 
                                      GlobalBlock(self.node_output_dim, self.graph_output_dim, hidden_dim=self.hidden_dim, output_dim=self.graph_output_dim, dropout=self.dropout, node_aggregation=self.aggregation)))



        self.classifier = Linear(self.node_output_dim, self.node_out_dim)
        self.logsoftmax = LogSoftmax(dim=1)
        self.sigmoid = Sigmoid()
        
    def forward(self, x, edge_index, edge_attr, u, batch):
        x, edge_attr, u = self.GN1(x, edge_index, edge_attr, u, batch)
        x, edge_attr, u = self.GN2(x, edge_index, edge_attr, u, batch)
        for i in range(self.hop - 2): x, edge_attr, u = self.GNn[i](x, edge_index, edge_attr, u, batch)

        if self.node_out_dim == 20: 
            out = self.logsoftmax(self.classifier(x))
        elif self.node_out_dim == 1: 
            out = self.sigmoid(self.classifier(x)).view(-1)
        else: raise NotImplementedError()
        
        return out, x, edge_attr, u




# The computational blocks in edge, node and global level
class EdgeBlock(torch.nn.Module):
    def __init__(self, node_attr_dim, edge_attr_dim, graph_attr_dim, \
                       hidden_dim=32, output_dim=32, dropout=0.2):
        
        super(EdgeBlock, self).__init__()
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.graph_attr_dim = graph_attr_dim
        self.dropout = dropout
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # print(self.node_attr_dim, self.edge_attr_dim, self.graph_attr_dim)
        self.input_dim = 2 * self.node_attr_dim + self.edge_attr_dim + self.graph_attr_dim
        
        self.edge_mlp = Sequential(Linear(self.input_dim, self.hidden_dim),
                                   ReLU(),
                                   BatchNorm1d(self.hidden_dim),
                                   Dropout(self.dropout),
                                   Linear(self.hidden_dim, self.output_dim))

        
    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        return self.edge_mlp(out)

    
class NodeBlock(torch.nn.Module):
    def __init__(self, node_attr_dim, edge_attr_dim, graph_attr_dim, \
                       hidden_dim=32, output_dim=32, dropout=0.2, aggregation='mean'):
        
        super(NodeBlock, self).__init__()
        self.node_attr_dim = node_attr_dim
        self.edge_attr_dim = edge_attr_dim
        self.graph_attr_dim = graph_attr_dim
        self.dropout = dropout
        self.aggregation = aggregation
        assert self.aggregation in ['mean', 'sum', 'max']
        
        self.hidden_dim = hidden_dim
        self.input_dim_mp = self.node_attr_dim + self.edge_attr_dim
        self.input_dim_mp2 = self.input_dim_mp + self.graph_attr_dim
        
        self.output_dim = output_dim
        
        self.node_mlp_1 = Sequential(Linear(self.input_dim_mp, self.hidden_dim), 
                                     ReLU(),
                                     BatchNorm1d(self.hidden_dim),
                                     Dropout(self.dropout),
                                     Linear(self.hidden_dim, self.output_dim))
        
        self.node_mlp_2 = Sequential(Linear(self.input_dim_mp2, self.hidden_dim), 
                                     ReLU(),
                                     BatchNorm1d(self.hidden_dim),
                                     Dropout(self.dropout),
                                     Linear(self.hidden_dim, self.output_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        
        if self.aggregation == 'mean': 
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        elif self.aggregation == 'sum': 
            out = scatter_sum(out, col, dim=0, dim_size=x.size(0))
        elif self.aggregation == 'max': 
            out = scatter_max(out, col, dim=0, dim_size=x.size(0))
        else: NotImplementedError()
        
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

    
class GlobalBlock(torch.nn.Module):
    
    def __init__(self, node_attr_dim, graph_attr_dim, \
                 hidden_dim=32, output_dim=32, dropout=0.2, node_aggregation='mean'):
        super(GlobalBlock, self).__init__()
        
        self.node_attr_dim = node_attr_dim
        # self.edge_attr_dim = edge_attr_dim
        self.graph_attr_dim = graph_attr_dim
        self.node_aggregation = node_aggregation
        # self.edge_aggregation = edge_aggregation
        assert self.node_aggregation in ['mean', 'sum', 'max'] 
        
        # self.input_dim = self.node_attr_dim + self.edge_attr_dim + self.graph_attr_dim
        self.input_dim = self.node_attr_dim + self.graph_attr_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        self.global_mlp = Sequential(Linear(self.input_dim, self.hidden_dim), 
                                     ReLU(),
                                     Dropout(self.dropout),
                                     Linear(self.hidden_dim, self.output_dim))

        
    def forward(self, x, edge_index, edge_attr, u, batch):
        
        # here we need to include the edge attributes as well
        # the \rho^{e -> u} function in Algorithm 1
        
        # aggrgate node features 
        if self.node_aggregation == 'mean': 
            agg_nodes = scatter_mean(x, batch, dim=0)
        elif self.node_aggregation == 'sum': 
            agg_nodes = scatter_sum(x, batch, dim=0)
        elif self.node_aggregation == 'max': 
            agg_nodes = scatter_max(x, batch, dim=0)
        else: NotImplementedError()
        
        # aggregate edge features
        # this will be expensive (?) and might not have insightful results
        # use edge batching and torch_scatter
        
        out = torch.cat([u, agg_nodes], dim=1)
        return self.global_mlp(out)



