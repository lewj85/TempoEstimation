import numpy as np


def binary_targets_to_beat_times(y, frame_rate):
    """
    Convert binary targets to beat times
    """
    beat_times = []
    for target in y:
        beat_times.append(np.nonzero(target)[0] / frame_rate)

    return beat_times


def estimate_beats_from_activation(beat_activation, frame_rate):
    # STUB
    pass
