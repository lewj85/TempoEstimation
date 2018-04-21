def construct_spectrogram_bilstm(nwin):
    a1 = Input(shape=(nwin, 513))
    a2 = Input(shape=(nwin, 1025))
    a3 = Input(shape=(nwin, 2049))
    a4 = Input(shape=(nwin, 513))
    a5 = Input(shape=(nwin, 1025))
    a6 = Input(shape=(nwin, 2049))
    alist = [a1,a2,a3,a4,a5,a6]
    a = keras.layers.Concatenate(axis=-1)(alist)
    output = bilstm(a)
    model = Model(inputs=alist, outputs=output)
    return model
