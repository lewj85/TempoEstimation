from keras.layers import Conv1D, Dense, Flatten, Input


def cnn(num_steps, win_size=1024, weight_decay=0.01):
    """
    Create a convolutional audio feature extraction architecture
    """
    a = Input(shape=(num_steps, win_size))
    filters = 256
    kernel_size = 128
    net = Conv1D(filters, kernel_size, strides=4, kernel_initializer='he_normal',
               kernel_regularizer=keras.regularizers.l2(weight_decay))(a)
    net = Conv1D(filters, kernel_size, strides=4, kernel_initializer='he_normal',
               kernel_regularizer=keras.regularizers.l2(weight_decay))(net)
    net = Conv1D(filters, kernel_size, strides=4, kernel_initializer='he_normal',
               kernel_regularizer=keras.regularizers.l2(weight_decay))(net)
    # remove channel dimension
    net = Flatten(data_format='channels_last')(net)

    net = Dense(units=128, activation='linear')(net)

    return net
