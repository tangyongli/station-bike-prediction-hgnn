import pickle
def save_hetero_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_hetero_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
