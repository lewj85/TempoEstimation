from keras.optimizers import Adam
from model.architectures import construct_spectrogram_bilstm
import numpy as np

def train_model(X, y, model_type, lr=0.0001, batch_size=5, num_epochs=10, valid_ratio=0.2, test_ratio=0.2):
    # create BiLSTM and dense
    if model_type == 'spectrogram':
        model = construct_spectrogram_bilstm(X[0].shape[1])
    else:
        raise ValueError('Unsupported model type: {}'.format(model_type))

    # Create computational graph
    adm = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adm)

    # Divide data
    n_examples = X.shape[0]
    # Get random order
    idx = np.random.permutation(n_examples)
    
    num_train = int(n_examples*(1-valid_ratio-test_ratio))
    num_valid = int(n_examples*valid_ratio)
    num_test = n_examples - num_train - num_valid
    
    train_idx = idx[:num_train]
    valid_idx = idx[num_train:num_train+num_valid]
    test_idx = idx[num_train+num_valid:]
    
    x = X[train_idx]
    y = Y[train_idx]
    x_valid = X[valid_idx]
    y_valid = y[valid_idx]
    x_test = X[test_idx]
    y_test = y[test_idx]
    
    # Training
    model.fit(x=X, y=y, validation_data=(x_valid, y_valid), epochs=num_epochs, batch_size=batch_size)
    
    # Run on test set
    model.predict(x_test)

    return model
