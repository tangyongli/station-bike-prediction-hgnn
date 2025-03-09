import pickle


def save_hetero_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_hetero_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Prediction result converts to non-normalization format
def denormalize(normalized_data, scaler_min, scaler_max):
    return normalized_data * (scaler_max - scaler_min + 1e-8) + scaler_min