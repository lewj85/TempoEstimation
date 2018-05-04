from keras.layers import Conv1D, Dense, Reshape, Input, TimeDistributed
import keras.regularizers


def cnn(num_steps, win_size=1024, weight_decay=0.01):
    """
    Create a convolutional audio feature extraction architecture
    """
    input_shape = (num_steps, win_size, 1)
    a = Input(shape=input_shape)
    filters = 128
    kernel_size = 128
    net = TimeDistributed(Conv1D(filters, kernel_size, strides=2, kernel_initializer='he_normal',
               kernel_regularizer=keras.regularizers.l2(weight_decay)), input_shape=input_shape)(a)
    net = TimeDistributed(Conv1D(filters, kernel_size, strides=2, kernel_initializer='he_normal',
               kernel_regularizer=keras.regularizers.l2(weight_decay)))(net)
    """
    net = TimeDistributed(Conv1D(filters, kernel_size, strides=1, kernel_initializer='he_normal',
               kernel_regularizer=keras.regularizers.l2(weight_decay)))(net)
    """
    # remove channel dimension
    net = Reshape((num_steps, -1))(net)

    net = TimeDistributed(Dense(units=128, activation='linear'))(net)

    return a, net
