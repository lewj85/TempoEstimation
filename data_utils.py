import numpy as np

def create_data_subsets(X, y, valid_ratio=0.2, test_ratio=0.2):
    """
    Split data into train, validation, and test subsets
    """
    # Divide data
    if type(X) == list:
        n_examples = X[0].shape[0]
    else:
        n_examples = X.shape[0]
    # Get random order
    idx = np.random.permutation(n_examples)

    num_train = int(n_examples*(1-valid_ratio-test_ratio))
    num_valid = int(n_examples*valid_ratio)
    num_test = n_examples - num_train - num_valid

    train_idx = idx[:num_train]
    valid_idx = idx[num_train:num_train+num_valid]
    test_idx = idx[num_train+num_valid:]

    train_data = {
        'X': X[train_idx],
        'y': y[train_idx],
        'indices': train_idx
    }

    valid_data = {
        'X': X[valid_idx],
        'y': y[valid_idx],
        'indices': valid_idx
    }

    test_data = {
        'X': X[test_idx],
        'y': y[test_idx],
        'indices': test_idx
    }

    return train_data, valid_data, test_data
