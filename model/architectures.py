from keras.models import Model
from keras.layers import concatenate, Input
from .bilstm import bilstm


def construct_spectrogram_bilstm(nwin):
    """
    Construct a spectrogram BiLSTM beat tracker model
    """
    # Make input layers for each spectrogram
    a1 = Input(shape=(nwin, 128))
    a2 = Input(shape=(nwin, 128))
    a3 = Input(shape=(nwin, 128))
    a4 = Input(shape=(nwin, 128))
    a5 = Input(shape=(nwin, 128))
    a6 = Input(shape=(nwin, 128))
    alist = [a1,a2,a3,a4,a5,a6]
    # Join the spectrograms into a single image
    a = concatenate(alist, axis=-1)
    # Pass spectrograms through BiLSTM
    output = bilstm(a, nwin, 128)
    model = Model(inputs=alist, outputs=output)
    return model
