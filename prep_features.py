import librosa
import numpy as np
from scipy.signal import medfilt


def prep_spectrogram_features(x, fs, hop_size=441):
    """
    Compute mel spectrogram and median spectrogram features,
    as is done in Bock et al. 2011
    """
    #x, fs = librosa.load(filename, sr=44100)

    # 3 melspectrograms
    s1 = librosa.core.stft(x, n_fft=1024, hop_length=hop_size, win_length=1024,
                           window='hamming', center=True, pad_mode='constant')
    ms1 = np.abs(librosa.feature.melspectrogram(sr=fs, S=s1))
    s2 = librosa.core.stft(x, n_fft=2048, hop_length=hop_size, win_length=2048,
                           window='hamming', center=True, pad_mode='constant')
    ms2 = np.abs(librosa.feature.melspectrogram(sr=fs, S=s2))
    s3 = librosa.core.stft(x, n_fft=4096, hop_length=hop_size, win_length=4096,
                           window='hamming', center=True, pad_mode='constant')
    ms3 = np.abs(librosa.feature.melspectrogram(sr=fs, S=s3))

    # 3 median spectrograms
    # pad edges
    p1 = np.pad(ms1, ((0,0),(0,1024//2)), 'reflect')
    p2 = np.pad(ms2, ((0,0),(0,2048//2)), 'reflect')
    p3 = np.pad(ms3, ((0,0),(0,4096//2)), 'reflect')
    # cut off the non-causal median leftover from the end
    m1 = medfilt(p1, kernel_size=(1,1024//100 - 1))[:, :-1024//2]
    m2 = medfilt(p2, kernel_size=(1,2048//100 - 1))[:, :-2048//2]
    m3 = medfilt(p3, kernel_size=(1,4096//100 - 1))[:, :-4096//2]
    # half-wave rectify
    h1 = np.maximum(ms1-m1,0)
    h2 = np.maximum(ms2-m2,0)
    h3 = np.maximum(ms3-m3,0)

    # transpose so time dimension is first
    return ms1.T, ms2.T, ms3.T, h1.T, h2.T, h3.T
