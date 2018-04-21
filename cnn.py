# CNN

from keras.layers import Conv1D
from keras.layers import Flatten

def cnn(x, num_steps, win_size=1024, weight_decay=0.01):

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
    
    net = Dense(units=128, activation='linear')

def fork (model, n=2):
    
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
        
    return forks


def bilstm(inputs):

    # fork the inputs
    left, right = fork(inputs)

    # LSTM left branch
    left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid'))
    left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid'))
    left.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid'))
    # LSTM right branch
    right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid', go_backwards=True))
    right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid', go_backwards=True))
    right.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
                   forget_bias_init='one', return_sequences=True, activation='tanh',
                   inner_activation='sigmoid', go_backwards=True))

    # combine
    output = Sequential()
    output.add(Merge([left, right], mode='concat'))
    # add dense at end - beat or no beat are the 2 units at end
    output.add(TimeDistributed(Dense(units=2, activation='softmax')))
    
    return output
