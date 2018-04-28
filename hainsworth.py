from scipy.io import loadmat
import os
import librosa
import resampy
from keras.preprocessing import sequence

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
    tempo_oddity_num = int(datarec[5][0][0][0])
    difficulty_num = int(datarec[6][0][0][0])
    subbeat_struct_num = int(datarec[7][0][0][0])
    tempo = float(datarec[8][0][0][0])
    num_beats = int(datarec[9][0][0][0])
    beats = datarec[10][0].flatten()

    #assert len(beats) == num_beats

    #y = beats[0].flatten()
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

def resample_hainsworth_beats(annotations, source_sr, target_sr):
    annotations['beats'] = ((target_sr/source_sr) * annotations['beats']).astype(int)

# wrapper for hainsworth data
def prep_hainsworth_data(data_dir, label_dir, target_sr=44100):
    data_array = []
    label_array = []

    for i in range(245):
        # {} for replacement, : format spec, 03 pad up to 3 leading 0s, d is int
        filename1 = "{:03d}.wav".format(i)
        filename2 = "{:03d}_info.mat".format(i)

        f1 = os.path.join(data_dir, filename1)
        f2 = os.path.join(label_dir, filename2)
        if os.path.exists(f1) and os.path.exists(f2):
            # Load audio at original sample rate
            x, sr = librosa.load(f1, sr=None)
            r = load_hainsworth_annotations(f2)
            resample_hainsworth_beats(r, sr, target_sr)

            # Resample audio to target sample rate
            if target_sr != sr:
                x = resampy.resample(x, sr, target_sr)

            data_array.append(x)
            label_array.append(r) # has tempos too

    return data_array, label_array
