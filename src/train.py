
import torch
from utils import metrics
import os
import logging

 
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def process_epoch(model, dataloader, train_dataset, hetero_data_example, batch_size, device, denormalize, criterion, optimizer=None):
    """
    Processes a single epoch (either training or evaluation).
    """
    total_loss = 0
    total_mse_original = 0
    batch_count = 0

    is_train = optimizer is not None 
    if is_train:
        model.train()
    else:
        model.eval()

    with torch.set_grad_enabled(is_train):  # Enable/disable gradients
        for batch in dataloader:
            if is_train:
                optimizer.zero_grad()
            x_dict, targets = batch
            targets = targets.to(device).permute(0, 2, 1)
            if targets.shape[0] != batch_size:
                continue
            # x_dict={key:value.to(device) for key,value in x_dict['features'].items()}
         
            # x_dict1={key:value for key,value in x_dict['edge_indices'].items()}
            predictions = model(x_dict)
            loss = criterion(predictions, targets)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

            y_true_orig = denormalize(targets, train_dataset.scaler['target']['min'],
                                      train_dataset.scaler['target']['max'])
            y_pred_orig = denormalize(predictions, train_dataset.scaler['target']['min'],
                                      train_dataset.scaler['target']['max'])
            mse_original = ((y_true_orig - y_pred_orig) ** 2).mean()
            total_mse_original += mse_original.item()
            batch_count += 1

    return total_loss / batch_count, total_mse_original / batch_count







# Function to denormalize
def denormalize(normalized_data, scaler_min, scaler_max):
    return normalized_data * (scaler_max - scaler_min + 1e-8) + scaler_min


def train_model(model, train_dataloader, val_dataloader, train_dataset,val_dataset, hetero_data_example, optimizer, criterion, modelname,best_model_path,checkpoint_path,args):
    """Trains the model, saves checkpoints, and  metrics."""
    best_val_mse = float('inf')
    start_epoch = 0
    train_losses, val_losses = [], []  # Initialize lists to store losses

    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_mse = checkpoint['best_val_mse']
        train_losses = checkpoint.get('train_losses', [])  # Load saved losses
        val_losses = checkpoint.get('val_losses', [])
        logging.info(f"Resuming training from epoch {start_epoch}")
    print(f"Resuming training from epoch {start_epoch}")
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
  
    # --- Early Stopping ---
    patience = 10  # Number of epochs to wait for improvement
    epochs_no_improve = 0  

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss, train_mse = process_epoch(model, train_dataloader, train_dataset, hetero_data_example,
                                               args.batch_size, args.device, denormalize, criterion, optimizer,
                                               )
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}")
        logging.info(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}")
        train_losses.append(train_loss)  # Append training loss

        val_loss, val_mse = process_epoch(model, val_dataloader, val_dataset, hetero_data_example,
                                           args.batch_size, args.device, denormalize, criterion,optimizer=None)
        logging.info(f"Epoch {epoch + 1}, Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}")
        
        val_losses.append(val_loss)      # Append validation loss
        scheduler.step(val_loss)
        # --- Save Metrics csv ----
        metrics.save_metrics(args.metricfilepath,modelname, epoch, train_mse, val_mse)
        # Save best model
        if val_mse < best_val_mse:
            # print('v',val_mse,best_val_mse)
            best_val_mse = val_mse
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved at epoch {epoch + 1} (Val MSE: {best_val_mse:.4f})")
            epochs_no_improve = 0  # Reset counter if validation loss improves
        else:
            epochs_no_improve += 1

        # Save checkpoint
        if (epoch + 1) % 1 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_mse': best_val_mse,
                'train_losses': train_losses,  
                'val_losses': val_losses,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved at epoch {epoch + 1}")

        # --- Early Stopping Check ---
        if epochs_no_improve == patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break