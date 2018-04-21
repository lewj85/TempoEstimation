hop_size = 1234
target_fs = 1234
batch_size = 5
num_epochs = 10

# prepare the data
a, r = prep_hainsworth_data(data_dir, label_dir, target_fs)
#a, r = prep_ballroom_data(data_dir, label_dir, target_fs)

# 6 spectrograms
X, y = preprocess_data(a, r, mode='spectrogram')

# create BiLSTM and dense
model = construct_spectrogram_bilstm(X[0].shape[0])

# create computational graph 
adm = Adam(lr=0.0001, decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=adm)
# training
model.fit(x=X, y=y, epochs=num_epochs, batch_size=batch_size)

# testing
