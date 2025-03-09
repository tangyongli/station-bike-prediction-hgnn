import os
import logging
import pandas as pd

def save_metrics(METRICS_FILE,model_name, epoch, train_mse, val_mse):
    """Saves training metrics to a CSV file."""
    try:
        if not os.path.exists(METRICS_FILE):
            df = pd.DataFrame(columns=['modelname', 'epoch', 'trainmse', 'valmse'])
        else:
            df = pd.read_csv(METRICS_FILE)

        new_row_data = [model_name, epoch, train_mse, val_mse]
        if len(new_row_data) != len(df.columns):
            raise ValueError(
                f"New row has {len(new_row_data)} elements, but DataFrame has {len(df.columns)} columns.")
        df.loc[len(df)] = new_row_data
        df.to_csv(METRICS_FILE, index=False)

    except (IndexError, ValueError) as e:
        logging.error(f"Error appending to DataFrame: {e}")
        

