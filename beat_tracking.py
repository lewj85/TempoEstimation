import numpy as np
from scipy.signal import lfilter
from math import ceil

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
    acr = np.correlate(beat_activation, beat_activation)
    
    # Smoothing
    acr_s = lfilter(np.hamming(int(0.15*frame_rate)), [1], acr)
    
    # Limit candidate range
    acr_lim = acr_s[lag_min:lag_max+1]
    
    # Select beat interval candidate (tau* or i) - frames per beat
    i = np.argmax(acr_lam) + lag_min
    
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
        local_beat_region = beat_activation[nk-d:nk+d]
        beat_locs[idx] = np.argmax(local_beat_region) + nk-d
    
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
    
    
    
    
    
    