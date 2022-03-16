
import argparse


parser = argparse.ArgumentParser(description="Interface GN Parameters")


# Task
parser.add_argument("--task", type=str, help="Task type", default='AA', choices=['AA', 'IF'])


# Model Parameters
parser.add_argument("--architecture", type=str, help="GNN Architecture", choices=['GCN', 'GAT', 'GraphSAGE', 'GraphNet'], required=True)

parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden Dimensions")
parser.add_argument("--node_out_dim", type=int, default=32, help="Node Out Dimensions")
parser.add_argument("--edge_out_dim", type=int, default=32, help="Edge Out Dimensions (for GN)")
parser.add_argument("--graph_out_dim", type=int, default=32, help="Graph Out Dimensions (for GN)")
parser.add_argument("--hop", type=int, default=3, help="Hop")
parser.add_argument("--dropout", type=float, default=0.25, help="Dropout (for GN)")
parser.add_argument("--heads", type=int, default=16, help="Attention heads (for GAT)")
parser.add_argument("--aggregation", type=str, default='mean', choices=['mean', 'sum', 'max'], help="Aggrgation type (for SAGE and GN)")




# Data parameters
parser.add_argument("--training_size", type=int, default=100, help="Training Size")
parser.add_argument("--validation_ratio", type=float, default=0.2, help="Validation Ratio")



# Training parameters
parser.add_argument("--batch_size", type=int, default=32, help="Minibatch Size")
parser.add_argument("--optimizer", type=str, default='AdamW', choices=['Adam', 'SGD', 'AdamW'], help="Optimizer")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
parser.add_argument("--log_every", type=int, default=50, help="Print log every")



# Other parameters
parser.add_argument("--log", action=argparse.BooleanOptionalAction)
parser.add_argument("--seed", type=int, default=99, help="Random seed")
parser.add_argument("--keep_model", type=bool, default=True, help="Save model")

