from scipy.io import loadmat
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

HAINSWORTH_STYLES = [
    'classical',
    'solo classical',
    'choral',
    'rock/pop',
    'dance',
    'unused',
    'jazz',
    'big band jazz',
    '60s pop',
    'folk',
    'random stuff'
]

HAINSWORTH_TEMPO_ODDITIES = [
    'normal',
    'rall',
    'sudden change',
    'rubato'
]

HAINSWORTH_SUBBEAT_STRUCTURES = [
    'not divided',
    'quavers',
    'semiquavers',
    'demisemiquavers',
    'triplets',
    'into 6',
    'swing',
    'swung semiquavers'
]

def load_hainsworth_annotations(filename):
    if not os.path.exists(filename):
        raise ValueError('{} does not exist'.format(filename))
    
    datarec = loadmat(filename)['datarec']
    filename = datarec[0][0][0]
    artist = datarec[1][0][0]
    title = datarec[2][0][0]
    duration = float(datarec[3][0][0][0])
    style_num = int(datarec[4][0][0][0])
    #print(style_num)
    tempo_oddity_num = int(datarec[5][0][0][0])
    difficulty_num = int(datarec[6][0][0][0])
    subbeat_struct_num = int(datarec[7][0][0][0])
    tempo = float(datarec[8][0][0][0])
    num_beats = int(datarec[9][0][0][0])
    beats = datarec[10][0].flatten()
    
    #assert len(beats) == num_beats
    

    y = beats[0].flatten()
    annotations = {
        'filename': filename,
        'artist': artist,
        'title': title,
        'duration': duration,
        'style': HAINSWORTH_STYLES[style_num - 1],
        'tempo_oddities': HAINSWORTH_TEMPO_ODDITIES[tempo_oddity_num - 1],
        'difficulty': difficulty_num,
        'subbeat_structure': HAINSWORTH_SUBBEAT_STRUCTURES[subbeat_struct_num - 1],
        'tempo': tempo,
        'beats': beats
    }
    return annotations

lags = []
tempos = []
label_dir = "./datasets/hainsworth/info/"
sample_dir = "./datasets/hainsworth/samples/"
for i in range(245):
    # {} for replacement, : format spec, 03 pad up to 3 leading 0s, d is int
    filename1 = "{:03d}_info.mat".format(i)
    filename2 = "{:03d}.wav".format(i)

    f1 = os.path.join(label_dir, filename1)
    f2 = os.path.join(sample_dir, filename2)
    if os.path.exists(f1) and os.path.exists(f2):
        # Load audio at original sample rate
        x, sr = librosa.load(f2, sr=None)
        r = load_hainsworth_annotations(f1)
        target_sr=44100
        lag = 60*target_sr / (441 * r['tempo'])
        tempos.append(r['tempo'])
        lags.append(lag)

#print(rs)
print(min(lags))
print(max(lags))
print(min(tempos))
print(max(tempos))

int_lags = [int(round(x)) for x in lags]
print(int_lags)
print(min(int_lags))
print(max(int_lags))

min_lag = 20
max_lag = 120

hainsworth_prior_histogram = np.zeros((int(max_lag+1-min_lag),))
#print(hainsworth_prior_histogram)

all_lags = list(range(min_lag, max_lag+1))
for i in int_lags:
    hainsworth_prior_histogram[i-min_lag] += 1
print(hainsworth_prior_histogram)
plt.bar(all_lags, hainsworth_prior_histogram)
