import os
from keras.optimizers import Adam
from model.callbacks import CSVLogger, EarlyStopping
from model.architectures import construct_spectrogram_bilstm
import numpy as np


def train_model(train_data, valid_data, model_type, model_output_dir, lr=0.0001,
                batch_size=5, num_epochs=10, patience=5):

    X_train = train_data['X']
    y_train = train_data['y']
    X_valid = valid_data['X']
    y_valid = valid_data['y']
    # create BiLSTM and dense
    if model_type == 'spectrogram':
        model = construct_spectrogram_bilstm(X_train[0].shape[1])
    else:
        raise ValueError('Unsupported model type: {}'.format(model_type))

    # Create computational graph
    adm = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adm)

    # Create callbacks
    cb = []
    # Add early stopping, which will terminate training if validation loss
    # does not improve after a number of epochs
    cb.append(EarlyStopping(monitor='val_loss', patience=patience))
    # Create a CSV of the training and validation metrics
    csv_path = os.path.join(model_output_dir, 'history.csv')
    cb.append(CSVLogger(csv_path))

    model_path = os.path.join(model_output_dir, 'model.hdf5')
    cb.append(ModelCheckpoint(model_path, save_best_only=True))

    # Training
    model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid),
              epochs=num_epochs, batch_size=batch_size, callbacks=cb, verbose=2)

    return model_path
