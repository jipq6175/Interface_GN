
import os
import random
import wandb

from datetime import datetime

from utils.data_utils import *
from utils.training_utils import train_GCN, train_Graph_Net, train_GAT, train_GraphSAGE
from utils.parser import parser





if __name__ == '__main__': 

    data_path = './data/raw_data/'
    save_path = './trained_models/'

    args = parser.parse_args()
    random.seed(args.seed)

    # task
    task = args.task
    classes = 20 if task == 'AA' else 1
    
    # model
    architecture = args.architecture
    hidden_dim = args.hidden_dim
    node_out_dim = args.node_out_dim
    edge_out_dim = args.edge_out_dim
    graph_out_dim = args.graph_out_dim
    hop = args.hop
    dropout = args.dropout
    heads = args.heads
    aggr = args.aggregation

    # data
    training_size = args.training_size
    validation_ratio = args.validation_ratio

    # training
    batch_size = args.batch_size
    optimizer_option = args.optimizer
    lr = args.learning_rate
    wd = args.weight_decay
    n_epochs = args.epochs

    # utils
    log = args.log
    keep_model = args.keep_model
    log_per_n = args.log_every


    # load data
    train_pdb_list = [os.path.join(data_path, x) for x in random.sample(os.listdir(data_path), training_size)]
    test_pdb_list = [os.path.join(data_path, x) for x in random.sample(os.listdir(data_path), int(training_size * validation_ratio))]

    train_dataloader = construct_dataloader_from_pdbs(train_pdb_list, batch_size=batch_size, shuffle=True, task=task)
    val_dataloader = construct_dataloader_from_pdbs(test_pdb_list, batch_size=batch_size, shuffle=False, task=task)

    # data = iter(train_dataloader).next()


    # initialize wandb
    now = datetime.now().strftime("%m-%d-%Y-%H-%M")
    project = 'Interface_GN'
    exp_name = f'{architecture}_{task}_{now}'
    
    if log: 
        wandb.login()
        wandb.init(project=project, 
                name=exp_name, 
                reinit=True,
            
                # Track hyperparameters and run metadata
                config={'learning_rate': lr,
                        'architecture': architecture,
                        'task': task,
                        'training_size': training_size,
                        'validation_ratio': validation_ratio,
                        'epochs': n_epochs,
                        'hidden_dim': hidden_dim,
                        'hop': hop,
                        'optimizer': optimizer_option,
                        'batch_size': batch_size,
                        'weight_decay': wd,
                        'dropout': dropout,
                        'heads': heads})

    
    if architecture == 'GCN': 
        trained_Model = train_GCN(train_dataloader, val_dataloader, n_epochs=n_epochs, log_per_n=log_per_n, \
                                  hidden_dim=hidden_dim, node_out_dim=classes, hop=hop, \
                                  optimizer_option=optimizer_option, lr=lr, log=log)
    

    elif architecture == 'GAT': 
        trained_Model = train_GAT(train_dataloader, val_dataloader, n_epochs=n_epochs, log_per_n=log_per_n, \
                                  hidden_dim=hidden_dim, node_out_dim=classes, hop=hop, heads=heads, \
                                  optimizer_option=optimizer_option, lr=lr, log=log)


    elif architecture == 'GraphSAGE': 
        trained_Model = train_GraphSAGE(train_dataloader, val_dataloader, n_epochs=n_epochs, log_per_n=log_per_n, \
                                        hidden_dim=hidden_dim, node_out_dim=classes, hop=hop, \
                                        optimizer_option=optimizer_option, lr=lr, log=log, aggr=aggr)


    elif architecture == 'GraphNet': 
        trained_Model = train_Graph_Net(train_dataloader, val_dataloader, n_epochs=n_epochs, log_per_n=log_per_n, \
                                        hidden_dim=hidden_dim, node_output_dim=node_out_dim, edge_output_dim=edge_out_dim, graph_output_dim=graph_out_dim, node_out_dim=classes, hop=hop, \
                                        dropout=dropout, aggregation=aggr, optimizer_option=optimizer_option, lr=lr, log=log)


    else: 
        raise NotImplementedError()


    if log: wandb.watch(trained_Model)
    print('Success !!')

    
    # dump the models into ./trained_models
    if keep_model: torch.save(trained_Model.state_dict(), os.path.join(save_path, exp_name + '_model.pt'))
    