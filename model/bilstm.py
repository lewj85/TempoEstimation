# Jason Cramer, Jesse Lew
from keras.layers import LSTM, Dense, TimeDistributed, concatenate, Input


def bilstm(inputs, nwin, feature_size, hidden_units=25):
    """
    Create a 3 layer BiLSTM archiecture with a binary decision layer, producing
    binary predictions per time-step
    """

    # LSTM left branch
    left = inputs
    left = LSTM(units=hidden_units, kernel_initializer='uniform',
                recurrent_initializer='uniform', return_sequences=True,
                activation='tanh', recurrent_activation='sigmoid')(left)#, input_shape=(nwin, feature_size))(left)
    left = LSTM(units=hidden_units, kernel_initializer='uniform',
                recurrent_initializer='uniform', return_sequences=True,
                activation='tanh', recurrent_activation='sigmoid')(left)
    left = LSTM(units=hidden_units, kernel_initializer='uniform',
                recurrent_initializer='uniform', return_sequences=True,
                activation='tanh', recurrent_activation='sigmoid')(left)
    # LSTM right branch
    right = inputs
    right = LSTM(units=hidden_units, kernel_initializer='uniform',
                 recurrent_initializer='uniform', return_sequences=True,
                 activation='tanh', recurrent_activation='sigmoid',
                 go_backwards=True)(right)#, input_shape=(nwin, feature_size))(right)
    right = LSTM(units=hidden_units, kernel_initializer='uniform',
                 recurrent_initializer='uniform', return_sequences=True,
                 activation='tanh', recurrent_activation='sigmoid',
                 go_backwards=True)(right)
    right = LSTM(units=hidden_units, kernel_initializer='uniform',
                 recurrent_initializer='uniform', return_sequences=True,
                 activation='tanh', recurrent_activation='sigmoid',
                 go_backwards=True)(right)

    # combine
    output = concatenate([left, right])
    # add dense at end - beat or no beat are the 2 units at end
    output = TimeDistributed(Dense(units=2, activation='softmax'))(output)#, input_shape=(nwin, 50))(output)

    return output
