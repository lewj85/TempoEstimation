from scipy.io import loadmat
import os
import librosa
import resampy
from tempo import get_interbeat_tempo_estimate



def load_ballroom_annotations(filename,sr):
    """
    Load Ballroom dataset annotations for a single file
    """

    if not os.path.exists(filename):
        raise ValueError('{} does not exist'.format(filename))

    f = open(filename, 'r')
    beats = []
    beat_times = []
    for line in f:
        t,b = line.strip().split()
        t = float(t)
        beats.append(int(t*sr))
        beat_times.append(t)


    annotations = {
        'filename': filename,
        'tempo': get_interbeat_tempo_estimate(beat_times),
        'beats': beats,
        'beat_times': beat_times
    }

    return annotations


# wrapper for ballroom data
def prep_ballroom_data(data_dir, label_dir, target_sr=44100, load_audio=True):
    """
    Load Ballroom dataset annotations and audio
    """
    data_array = []
    label_array = []

    w = os.walk(data_dir)
    for root, dirnames, filenames in w:
        for f in filenames:
            if str(f)[-4:] == ".wav":
                filename1 = str(f)
                filename2 = filename1[:-3]+str("beats")

                f1 = os.path.join(root, filename1)
                f2 = os.path.join(label_dir, filename2)
                if os.path.exists(f1) and os.path.exists(f2):
                    # Load audio at original sample rate
                    x, sr = librosa.load(f1, sr=None)
                    r = load_ballroom_annotations(f2,sr)
                    #resample_ballroom_beats(r, sr, target_sr)

                    # Resample audio to target sample rate
                    if load_audio:
                        if target_sr != sr:
                            x = resampy.resample(x, sr, target_sr)

                        data_array.append(x)
                    label_array.append(r)

    return data_array, label_array
