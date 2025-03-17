
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import warnings

from data import dataset_generator
from config import get_args
from utils import data_utils
import train
from models import h_tgnn
# time.sleep(1800)
warnings.filterwarnings("ignore", category=UserWarning)




def main(args: argparse.Namespace):
    """Main function to run the training and evaluation."""
    
    # ----- Feature preparation---------
    node_dynamicfeatures_optimized=np.load('Data/processed_data/inputsarrayforTGNN_dynamicfeatures(flowsweatherhourday).npy') # (5833, 1615, 8)
    hetero_data_example = data_utils.load_hetero_data('Data/processed_data/hetero_data_addcensus1.pkl')
    # print(hetero_data_example.metadata())
    train_dataset,val_dataset,test_dataset,train_dataloader, val_dataloader, test_dataloader=dataset_generator.create_data_loaders(node_dynamicfeatures_optimized, hetero_data_example,args)

   
   
    # -----Training---------
    torch.manual_seed(42) 
    if args.modelname=='GRU':
        model = h_tgnn.GRU(args.hidden_channels, args.hidden_units_gru,args.prediction_length)
    if args.modelname=='homog-gru': 
       model=h_tgnn.TGCN(11,args.hidden_channels, args.hidden_units_gru,args.prediction_length)
    if args.modelname=='hetero-gru': 
        model = h_tgnn.HeteroGNN(args.hidden_channels, args.hidden_units_gru,args.prediction_length)
    model = model.to(args.device)
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    train.train_model(model, train_dataloader, val_dataloader, train_dataset,val_dataset, optimizer, criterion, args.modelname,args.best_model_path,args.checkpoint_path,args)
    
        
    

if __name__ == "__main__":
    args = get_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main(args)