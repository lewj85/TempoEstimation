import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from prep_features import prep_spectrogram_features


def preprocess_data(audio_array, annotation_array, hop_size=441, mode='spectrogram', sr=44100):
    """
    Preprocess audio and annotation data into input features and targets
    """
    if mode == 'spectrogram':
        # Compute spectrogram features
        X = [pad_sequences(feat) for feat in zip(*[prep_spectrogram_features(x, sr) for x in audio_array])]
        max_len = X[0].shape[1]
    elif mode == 'audio':
        # Compute audio features
        # TODO: Chunk into frames
        X = pad_sequences(audio_array)
        max_len = X.shape[1]

    # Create targets for positive class
    y = pad_sequences([to_categorical((np.array(ann['beats']) / hop_size).astype(int),
                                      num_classes=max_len).sum(axis=0)
                       for ann in annotation_array])[:,:,np.newaxis]
    # Create targets for negative class and concatenate
    y = np.concatenate([(1-y), y], axis=-1)

    return X, y
