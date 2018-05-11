# Jason Cramer, Jesse Lew
import numpy as np
from scipy.signal import lfilter
from math import ceil
from filter_utils import lfilter_center


def get_beat_times_from_annotations(annotation_array, idxs):
    """
    Get a list of beat times from annotations
    """
    beat_times = []
    for idx in idxs:
        beat_times.append(annotation_array[idx]['beat_times'])

    return beat_times


def binary_targets_to_beat_times(y, frame_rate):
    """
    Convert binary targets to beat times
    """
    beat_times = []
    for target in y:
        beat_times.append(np.nonzero(target)[0] / frame_rate)

    return beat_times


def estimate_beats_from_activation(beat_activation, frame_rate, lag_min, lag_max, deviation=0.1):
    """
    Estimates beats in seconds using output from the NN
    """
    # Autocorrelation
    acr = np.correlate(beat_activation, beat_activation, mode='full')

    # Only take positive lags
    acr = acr[acr.size//2:]

    # Smoothing
    acr_s = lfilter_center(np.hamming(int(0.15*frame_rate)), acr)

    # Limit candidate range
    acr_lim = acr_s[lag_min:lag_max+1]

    # Select beat interval candidate (tau* or i) - frames per beat
    i = np.argmax(acr_lim) + lag_min

    # Find the frame of the first beat given the lag candidate (p*)
    frame_sums = []
    for p in range(i+1):
        frame_sums.append(np.sum(beat_activation[np.arange(p,len(beat_activation), step=i)]))
    beat_phase = np.argmax(frame_sums)

    # Find local peaks around beat cand locations - move through at step=i
    d = ceil(deviation * i)
    nklist = np.arange(beat_phase,len(beat_activation), step=i)
    beat_locs = np.zeros((len(nklist),))
    for idx,nk in enumerate(nklist):
        start_idx = max(0, nk-d)
        end_idx = min(nk+d, len(beat_activation))
        local_beat_region = beat_activation[start_idx:end_idx]
        beat_locs[idx] = np.argmax(local_beat_region) + start_idx

    # Divide by framerate to get values in seconds
    beat_locs /= frame_rate

    return beat_locs


def estimate_beats_for_batch(y_pred, frame_rate, lag_min, lag_max, deviation=0.1):
    """
    Estimates beats in seconds using *all* the outputs from the NN
    """
    all_beats = []
    for track in y_pred:
        all_beats.append(estimate_beats_from_activation(track, frame_rate, lag_min, lag_max, deviation))

    return all_beats
