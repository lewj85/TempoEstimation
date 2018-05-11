# Jason Cramer, Jesse Lew
import numpy as np

def create_data_subsets(X, y, valid_ratio=0.2):
    """
    Split data into train and validation subsets
    """
    # If we have a single input, wrap it in a list so we can generalize to
    # multiple inputs
    if type(X) == np.ndarray:
        X = [X]

    # Divide data
    n_examples = X[0].shape[0]
    # Get random order
    idx = np.random.permutation(n_examples)

    num_train = int(n_examples*(1-valid_ratio))
    num_valid = n_examples - num_train

    train_idx = idx[:num_train]
    valid_idx = idx[num_train:]

    train_data = {
        'X': [x[train_idx] for x in X],
        'y': y[train_idx],
        'indices': train_idx
    }

    valid_data = {
        'X': [x[valid_idx] for x in X],
        'y': y[valid_idx],
        'indices': valid_idx
    }

    # If we only have one input, remove the wrapper input list
    if len(X) == 1:
        train_data['X'] = train_data['X'][0]
        valid_data['X'] = valid_data['X'][0]

    return train_data, valid_data


def load_data(data_path, model_type):
    """
    Load model input data
    """
    data = dict(np.load(data_path))

    if model_type == 'spectrogram':
        data['X'] = [x for x in data['X']]

    return data
