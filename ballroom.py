from scipy.io import loadmat
import os
import librosa
import resampy
from keras.preprocessing import sequence

def load_ballroom_annotations(filename,sr):

    if not os.path.exists(filename):
        raise ValueError('{} does not exist'.format(filename))

    assert len(beats) == num_beats

    f = open(filename, 'r')
    beats = []
    beat_times = []
    for line in f:
        t,b = line.split(" ")
        beats.append(int(t*sr))
        beat_times.append(t)

    annotations = {
        'filename': filename,
#    'album_or_media': album_or_media,
#    'album_title': album_title,
#    'title': title,
#    'style': style,
#    'tempo': tempo,
        'beats': beats,
        'beat_times': beat_times
    }

    return annotations

#def resample_ballroom_beats(annotations, source_sr, target_sr):
#    annotations['beats'] = ((target_sr/source_sr) * annotations['beats']).astype(int)

# wrapper for ballroom data
def prep_ballroom_data(data_dir, label_dir, target_sr=44100, load_audio=True):
    data_array = []
    label_array = []

    w = os.walk(data_dir)
    for d in w:
        a,b,c = d
        for f in c:
            if str(f)[-4:] == ".wav":
                filename1 = str(f)
                filename2 = filename1[:-3]+str("beats")

                f1 = os.path.join(data_dir, filename1)
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
