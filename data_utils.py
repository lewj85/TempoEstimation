import numpy as np

def create_data_subsets(X, y, valid_ratio=0.2, test_ratio=0.2):
    """
    Split data into train, validation, and test subsets
    """
    # If we have a single input, wrap it in a list so we can generalize to
    # multiple inputs
    if type(X) == np.ndarray:
        X = [X]

    # Divide data
    n_examples = X[0].shape[0]
    # Get random order
    idx = np.random.permutation(n_examples)

    num_train = int(n_examples*(1-valid_ratio-test_ratio))
    num_valid = int(n_examples*valid_ratio)
    num_test = n_examples - num_train - num_valid

    train_idx = idx[:num_train]
    valid_idx = idx[num_train:num_train+num_valid]
    test_idx = idx[num_train+num_valid:]

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

    test_data = {
        'X': [x[test_idx] for x in X],
        'y': y[test_idx],
        'indices': test_idx
    }

    # If we only have one input, remove the wrapper input list
    if len(X) == 1:
        train_data['X'] = train_data['X'][0]
        valid_data['X'] = valid_data['X'][0]
        test_data['X'] = test_data['X'][0]

    return train_data, valid_data, test_data
