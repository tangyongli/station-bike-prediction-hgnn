
import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--modelname', type=str, default='hetero-gru')
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='Number of hidden channels in GNN layers.')
    parser.add_argument('--hidden_units_gru', type=int, default=32,
                        help='Number of hidden units in GRU layer.')
    parser.add_argument('--sequence_length', type=int, default=6,
                        help='sequence_length.')
    parser.add_argument('--prediction_length', type=int, default=3,
                        help='prediction_length')
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = parser.parse_args([])  
    args.checkpoint_path = f'Log/models/checkpoint/checkpoint_{args.modelname}_{args.hidden_channels}_{args.hidden_units_gru}.pt'
    args.best_model_path = f'Log/models/bestmodel/bestmodel_{args.modelname}_{args.hidden_channels}_{args.hidden_units_gru}.pt'
    args.metricfilepath=f'Log/metric_{args.modelname}_{args.hidden_channels}_{args.hidden_units_gru}.csv'
    return args

args = get_args()
