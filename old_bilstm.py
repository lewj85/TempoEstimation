# BiLSTM

#from random import random
#from numpy import array
#from numpy import cumsum
#import sklearn
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
#from keras.layers import Bidirectional
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Dense

filename = 'asdf'

inputs = prep_features(filename)
tempos = blah


# inputs and outputs
nfft = 
nwin = 
a1 = Input(shape=(nwin,nfft/2 + 1))
a2 = Input(shape=(nwin,nfft/2 + 1))
a3 = Input(shape=(nwin,nfft/2 + 1))
a4 = Input(shape=(nwin,nfft/2 + 1))
a5 = Input(shape=(nwin,nfft/2 + 1))
a6 = Input(shape=(nwin,nfft/2 + 1))
alist = [a1,a2,a3,a4,a5,a6]
a = keras.layers.Concatenate(axis=-1)(alist)

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks

# fork the inputs
left, right = fork(a)

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

# stochastic gradient descent vs ADAM
adm = Adam(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True)

# create computational graph 
model = Model(inputs=alist, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer=adm)

print("Train...")
model.fit(X_train, Y_train, batch_size=1, nb_epoch=nb_epoches, validation_data=(X_test, Y_test), verbose=1, show_accuracy=True)





