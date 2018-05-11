# Jason Cramer, Jesse Lew
import numpy as np
from scipy.signal import lfilter


def lfilter_center(b, x, pad_mode='constant'):
    """
    Applies centered (and thus non-causal) FIR filtering to the given signal
    """
    N = len(x)
    pad_length = len(b) // 2
    x_pad = np.pad(x, (0, pad_length), mode=pad_mode)
    res_pad = lfilter(b, [1], x_pad)
    return res_pad[pad_length:]
