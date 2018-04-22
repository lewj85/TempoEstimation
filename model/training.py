from keras.optimizers import Adam
from model.architectures import construct_spectrogram_bilstm


def train_model(X, y, model_type, lr=0.0001, batch_size=5, num_epochs=10):
    # create BiLSTM and dense
    if model_type == 'spectrogram':
        model = construct_spectrogram_bilstm(X[0].shape[1])
    else:
        raise ValueError('Unsupported model type: {}'.format(model_type))

    # create computational graph
    adm = Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adm)

    # training
    model.fit(x=X, y=y, epochs=num_epochs, batch_size=batch_size)

    return model
