def preprocess_data(audio_array, label_array, hop_size=441, mode='spectrogram', sr=44100):

    if mode == 'spectrogram':
        # magic
        X = tuple(sequence(feat) for feat in zip(*[prep_features(x, sr) for x in audio_array]))
    elif mode == 'audio':
        X = sequence(audio_array)
        
    y = sequence([keras.utils.to_categorical((labels / hop_size).astype(int), num_classes=len(labels))
                  for labels in label_array])
    
    return X, y
