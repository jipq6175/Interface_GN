# training utilities for different models

# given some train_dataloader and test_dataloader
# and some hyperparameters 


import torch
import wandb
import numpy as np
from tqdm.auto import tqdm
from model.gcn import GCN
from model.gat import GAT
from model.GraphSAGE import GraphSAGE
from model.graph_net import GraphNet

from torch.optim import SGD, Adam, AdamW



dv = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n\nUsing {dv} device\n")
DEVICE = torch.device(dv)


# get minibatch accuracies
def get_mini_batch_accuracy(batch_data, out, task='AA'): 
    if task == 'AA': 
        n_total = batch_data.node_mask.sum().detach().numpy()
        n_correct = torch.eq(batch_data.y[batch_data.node_mask], torch.argmax(out[batch_data.node_mask], dim=1)).sum().detach().numpy()
    elif task == 'IF': 
        n_total = batch_data.num_nodes
        n_correct = torch.eq(batch_data.y[batch_data.node_mask], out[batch_data.node_mask] >= 0.5).sum().detach().numpy()
    return n_correct, n_total


# wandb logging
def wandb_logging(loss, train_accuracy, val_accuracy): 
    wandb.log({'loss': loss, \
               'train_accuracy': train_accuracy, \
               'val_accuracy': val_accuracy})
    return None


# wandb summary to the final table
def wandb_summary(loss, train_accuracy, val_accuracy): 
    wandb.summary['loss'] = loss
    wandb.summary['train_accuracy'] = train_accuracy
    wandb.summary['val_accuracy'] =  val_accuracy
    return None


# Train GCN and return the trained model
# Todo: add wandb logger
def train_GCN(train_dataloader, val_dataloader, n_epochs=1000, log_per_n=50, \
              hidden_dim=64, node_out_dim=20, hop=3, \
              optimizer_option='AdamW', lr=1e-3, weight_decay=0.01, \
              log=True): 

    assert optimizer_option in ['AdamW', 'SGD', 'Adam']
    assert node_out_dim in [20, 1]
    task = 'AA' if node_out_dim == 20 else 'IF'

    batch_data = iter(train_dataloader).next()
    node_attr_dim = batch_data.x.shape[1]

    # initialize the model
    GCNModel = GCN(node_attr_dim, node_out_dim=node_out_dim, hidden_dim=hidden_dim, hop=hop)

    # criterion and optimizer
    if optimizer_option == 'AdamW': optimizer = AdamW(GCNModel.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_option == 'Adam': optimizer = Adam(GCNModel.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_option == 'SGD': optimizer = SGD(GCNModel.parameters(), lr=lr, weight_decay=weight_decay)
    else: raise NotImplementedError()
    
    if node_out_dim == 20: criterion = torch.nn.NLLLoss() 
    elif node_out_dim == 1: criterion = torch.nn.BCELoss()
    else: raise NotImplementedError()

    losses, train_accuracies, val_accuracies = [], [], []
    
    # epoch loop
    for epoch in tqdm(range(n_epochs + 1), desc='Training'): 

        GCNModel.train()
        train_correct, train_total, val_correct, val_total, minibatch_losses = [], [], [], [], []

        # training dataloader loop
        for train_data in train_dataloader: 
            
            optimizer.zero_grad()
            out, node_embeddings = GCNModel(train_data.x, train_data.edge_index)
            loss = criterion(out[train_data.node_mask], train_data.y[train_data.node_mask])
            loss.backward()
            optimizer.step()

            minibatch_losses.append(loss.detach().numpy())
            correct, total = get_mini_batch_accuracy(train_data, out, task=task)
            train_correct.append(correct)
            train_total.append(total)

        losses.append(np.array(minibatch_losses).mean())
        train_accuracies.append(np.array(train_correct).sum() / np.array(train_total).sum())


        # val dataloader loop
        GCNModel.eval()
        for val_data in val_dataloader: 
            pred, __ = GCNModel(val_data.x, val_data.edge_index)
            correct, total = get_mini_batch_accuracy(val_data, pred, task=task)
            val_correct.append(correct)
            val_total.append(total)
        
        val_accuracies.append(np.array(val_correct).sum() / np.array(val_total).sum())

        # wandb logging
        if log: wandb_logging(losses[-1], train_accuracies[-1], val_accuracies[-1])
            
        # reporter
        if epoch % log_per_n == 0:
            print(f'epoch = {epoch}: NLL Loss = {losses[-1]:.2f}, Train Acc = {train_accuracies[-1]:.2f}, Val Acc = {val_accuracies[-1]:.2f}')

    if log: wandb_summary(losses[-1], train_accuracies[-1], val_accuracies[-1])
    return GCNModel




# Training GAT
def train_GAT(train_dataloader, val_dataloader, n_epochs=1000, log_per_n=50, \
              hidden_dim=64, node_out_dim=20, heads=1, hop=3, \
              optimizer_option='AdamW', lr=1e-3, weight_decay=0.01, \
              log=True): 

    assert optimizer_option in ['AdamW', 'SGD', 'Adam']
    assert node_out_dim in [20, 1]
    task = 'AA' if node_out_dim == 20 else 'IF'

    batch_data = iter(train_dataloader).next()
    node_attr_dim = batch_data.x.shape[1]
    edge_attr_dim = batch_data.edge_attr.shape[1]

    # initialize the model
    GATModel = GAT(node_attr_dim, edge_attr_dim, node_out_dim=node_out_dim, hidden_dim=hidden_dim, hop=hop, heads=heads)

    # criterion and optimizer
    if optimizer_option == 'AdamW': optimizer = AdamW(GATModel.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_option == 'Adam': optimizer = Adam(GATModel.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_option == 'SGD': optimizer = SGD(GATModel.parameters(), lr=lr, weight_decay=weight_decay)
    else: raise NotImplementedError()
    
    if node_out_dim == 20: criterion = torch.nn.NLLLoss() 
    elif node_out_dim == 1: criterion = torch.nn.BCELoss()
    else: raise NotImplementedError()

    losses, train_accuracies, val_accuracies = [], [], []
    
    # epoch loop
    for epoch in tqdm(range(n_epochs + 1), desc='Training'): 

        GATModel.train()
        train_correct, train_total, val_correct, val_total, minibatch_losses = [], [], [], [], []

        # training dataloader loop
        for train_data in train_dataloader: 
            
            optimizer.zero_grad()
            out, node_embeddings = GATModel(train_data.x, train_data.edge_index, train_data.edge_attr)
            loss = criterion(out[train_data.node_mask], train_data.y[train_data.node_mask])
            loss.backward()
            optimizer.step()

            minibatch_losses.append(loss.detach().numpy())
            correct, total = get_mini_batch_accuracy(train_data, out, task=task)
            train_correct.append(correct)
            train_total.append(total)

        losses.append(np.array(minibatch_losses).mean())
        train_accuracies.append(np.array(train_correct).sum() / np.array(train_total).sum())


        # val dataloader loop
        GATModel.eval()
        for val_data in val_dataloader: 
            pred, __ = GATModel(val_data.x, val_data.edge_index, val_data.edge_attr)
            correct, total = get_mini_batch_accuracy(val_data, pred, task=task)
            val_correct.append(correct)
            val_total.append(total)
        
        val_accuracies.append(np.array(val_correct).sum() / np.array(val_total).sum())

        # wandb logging
        if log: wandb_logging(losses[-1], train_accuracies[-1], val_accuracies[-1])

        # reporter
        if epoch % log_per_n == 0:
            print(f'epoch = {epoch}: NLL Loss = {losses[-1]:.2f}, Train Acc = {train_accuracies[-1]:.2f}, Val Acc = {val_accuracies[-1]:.2f}')
    if log: wandb_summary(losses[-1], train_accuracies[-1], val_accuracies[-1]) 
    return GATModel


# training GraphSAGE
def train_GraphSAGE(train_dataloader, val_dataloader, n_epochs=1000, log_per_n=50, \
                    hidden_dim=64, node_out_dim=20, hop=3, aggr='mean', \
                    optimizer_option='AdamW', lr=1e-3, weight_decay=0.01, \
                    log=True): 

    assert optimizer_option in ['AdamW', 'SGD', 'Adam']
    assert node_out_dim in [20, 1]
    task = 'AA' if node_out_dim == 20 else 'IF'

    batch_data = iter(train_dataloader).next()
    node_attr_dim = batch_data.x.shape[1]

    # initialize the model
    SAGEModel = GraphSAGE(node_attr_dim, node_out_dim=node_out_dim, hidden_dim=hidden_dim, hop=hop, aggr=aggr)

    # criterion and optimizer
    if optimizer_option == 'AdamW': optimizer = AdamW(SAGEModel.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_option == 'Adam': optimizer = Adam(SAGEModel.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_option == 'SGD': optimizer = SGD(SAGEModel.parameters(), lr=lr, weight_decay=weight_decay)
    else: raise NotImplementedError()
    
    if node_out_dim == 20: criterion = torch.nn.NLLLoss() 
    elif node_out_dim == 1: criterion = torch.nn.BCELoss()
    else: raise NotImplementedError()

    losses, train_accuracies, val_accuracies = [], [], []
    
    # epoch loop
    for epoch in tqdm(range(n_epochs + 1), desc='Training'): 

        SAGEModel.train()
        train_correct, train_total, val_correct, val_total, minibatch_losses = [], [], [], [], []

        # training dataloader loop
        for train_data in train_dataloader: 
            
            optimizer.zero_grad()
            out, node_embeddings = SAGEModel(train_data.x, train_data.edge_index)
            loss = criterion(out[train_data.node_mask], train_data.y[train_data.node_mask])
            loss.backward()
            optimizer.step()

            minibatch_losses.append(loss.detach().numpy())
            correct, total = get_mini_batch_accuracy(train_data, out, task=task)
            train_correct.append(correct)
            train_total.append(total)

        losses.append(np.array(minibatch_losses).mean())
        train_accuracies.append(np.array(train_correct).sum() / np.array(train_total).sum())


        # val dataloader loop
        SAGEModel.eval()
        for val_data in val_dataloader: 
            pred, __ = SAGEModel(val_data.x, val_data.edge_index)
            correct, total = get_mini_batch_accuracy(val_data, pred, task=task)
            val_correct.append(correct)
            val_total.append(total)
        
        val_accuracies.append(np.array(val_correct).sum() / np.array(val_total).sum())

        if log: wandb_logging(losses[-1], train_accuracies[-1], val_accuracies[-1])

        # reporter
        if epoch % log_per_n == 0:
            print(f'epoch = {epoch}: NLL Loss = {losses[-1]:.2f}, Train Acc = {train_accuracies[-1]:.2f}, Val Acc = {val_accuracies[-1]:.2f}')
    if log: wandb_summary(losses[-1], train_accuracies[-1], val_accuracies[-1])
    return SAGEModel





# train graph nets
def train_Graph_Net(train_dataloader, val_dataloader, n_epochs=1000, log_per_n=50, \
                    hidden_dim=64, node_output_dim=32, edge_output_dim=32, graph_output_dim=32, node_out_dim=20, hop=3, dropout=0.2, aggregation='mean', \
                    optimizer_option='AdamW', lr=1e-3, weight_decay=0.01, \
                    log=True): 

    assert optimizer_option in ['AdamW', 'SGD', 'Adam']
    assert node_out_dim in [20, 1]
    task = 'AA' if node_out_dim == 20 else 'IF'

    batch_data = iter(train_dataloader).next()
    edge_attr_dim = batch_data.edge_attr.shape[1]
    node_attr_dim = batch_data.x.shape[1]
    graph_attr_dim = batch_data.u.shape[1]
    

    # initialize the model
    GraphNetNModel = GraphNet(node_attr_dim, edge_attr_dim, graph_attr_dim, \
                              hidden_dim=hidden_dim, node_output_dim=node_output_dim, edge_output_dim=edge_output_dim, graph_output_dim=graph_output_dim, node_out_dim=node_out_dim,\
                              dropout=dropout, aggregation=aggregation, hop=hop)

    # criterion and optimizer
    if optimizer_option == 'AdamW': optimizer = AdamW(GraphNetNModel.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_option == 'Adam': optimizer = Adam(GraphNetNModel.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_option == 'SGD': optimizer = SGD(GraphNetNModel.parameters(), lr=lr, weight_decay=weight_decay)
    else: raise NotImplementedError()
    
    if node_out_dim == 20: criterion = torch.nn.NLLLoss() 
    elif node_out_dim == 1: criterion = torch.nn.BCELoss()
    else: raise NotImplementedError()

    losses, train_accuracies, val_accuracies = [], [], []
    
    # epoch loop
    for epoch in tqdm(range(n_epochs + 1), desc='Training'): 

        GraphNetNModel.train()
        train_correct, train_total, val_correct, val_total, minibatch_losses = [], [], [], [], []

        # training dataloader loop
        for train_data in train_dataloader: 
            
            optimizer.zero_grad()
            out, node_embeddings, edge_embeddings, graph_embeddings = \
                GraphNetNModel(train_data.x, train_data.edge_index, train_data.edge_attr, train_data.u, train_data.batch)
            
            loss = criterion(out[train_data.node_mask], train_data.y[train_data.node_mask])
            loss.backward()
            optimizer.step()

            minibatch_losses.append(loss.detach().numpy())
            correct, total = get_mini_batch_accuracy(train_data, out, task=task)
            train_correct.append(correct)
            train_total.append(total)

        losses.append(np.array(minibatch_losses).mean())
        train_accuracies.append(np.array(train_correct).sum() / np.array(train_total).sum())


        # val dataloader loop
        GraphNetNModel.eval()
        for val_data in val_dataloader: 
            pred, __, __, __ = GraphNetNModel(val_data.x, val_data.edge_index, val_data.edge_attr, val_data.u, val_data.batch)
            correct, total = get_mini_batch_accuracy(val_data, pred, task=task)
            val_correct.append(correct)
            val_total.append(total)
        
        val_accuracies.append(np.array(val_correct).sum() / np.array(val_total).sum())

        # wandb log
        if log: wandb_logging(losses[-1], train_accuracies[-1], val_accuracies[-1])

        # reporter
        if epoch % log_per_n == 0:
            print(f'epoch = {epoch}: NLL Loss = {losses[-1]:.2f}, Train Acc = {train_accuracies[-1]:.2f}, Val Acc = {val_accuracies[-1]:.2f}')
    if log: wandb_summary(losses[-1], train_accuracies[-1], val_accuracies[-1])
    return GraphNetNModel



