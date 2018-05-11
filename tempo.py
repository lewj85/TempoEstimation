# Jason Cramer, Jesse Lew
import numpy as np
import scipy.signal as signal
from filter_utils import lfilter_center


def get_interbeat_tempo_estimate(beat_times):
    """
    Get tempo estimate derived from the median interbeat interval
    """
    return 60 / np.median(np.diff(np.sort(beat_times)))


def get_tempos_from_annotations(annotation_array, idxs):
    """
    Get a list of tempos from annotations
    """
    tempos = []
    for idx in idxs:
        tempos.append(annotation_array[idx]['tempo'])

    return tempos


def get_comb_filter_coeffs(alpha, lag):
    """
    Get filter coefficients for a resonating comb filter bank
    """
    b = [1]
    a = [1] + [0] * (lag-1) + [-alpha]

    return b, a


def estimate_tempo(beat_act, frame_rate, lag_min, lag_max, num_tempo_steps,
                   alpha, smooth_win_len, tempo_prior):
    """
    Estimate tempo using a filterbank of resonating comb filters and a weighted histogram
    """

    # Smooth beat activation
    win = np.hamming(int(np.ceil(smooth_win_len*frame_rate)))
    beat_act_smooth = signal.lfilter(win, [1], beat_act)

    # Get lag range
    lags = np.arange(lag_min, lag_max+1)

    # Get filter bank output
    filterbank_output = []
    for lag in lags:
        b, a = get_comb_filter_coeffs(alpha, lag)
        filterbank_output.append(signal.lfilter(b, a, beat_act_smooth))
    filterbank_output = np.vstack(filterbank_output)

    # Construct weighted histogram
    hist = np.zeros((len(lags),))
    for time_idx, lag_idx in enumerate(filterbank_output.argmax(axis=0)):
        hist[lag_idx] += filterbank_output[lag_idx, time_idx]

    # Smooth histogram by convolving with a hamming window
    hist_win = np.hamming(7)
    hist_smooth = lfilter_center(hist_win, hist)

    if tempo_prior is not None:
        hist_smooth *= tempo_prior

    # Get the peak lag and get the corresponding tempo in beats per minute
    lag_est = lags[hist_smooth.argmax()]
    bpm_est = 60 * frame_rate / lag_est

    return bpm_est


def estimate_tempos_for_batch(y_pred, frame_rate, lag_min, lag_max,
    num_tempo_steps=100, alpha=0.79, smooth_win_len=.14, tempo_prior=None):
    """
    Estimate tempos for a batch of outputs
    """
    tempos = []
    for track in y_pred:
        tempos.append(estimate_tempo(track, frame_rate, lag_min, lag_max,
                                     num_tempo_steps, alpha, smooth_win_len,
                                     tempo_prior=tempo_prior))

    return np.array(tempos)
