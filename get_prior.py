from scipy.signal import gaussian, filtfilt #, lfilter
import numpy as np
import matplotlib.pyplot as plt

def get_hainsworth_prior(r, sr, target_sr=44100, hop_size=441, k=1, show=False)
    """
    Gets prior histogram of tempos from the Hainsworth dataset.
    """

    lags = []
    tempos = []

    for i in range(len(r)):
        lag = 60*target_sr / (hop_size * r[i]['tempo'])
        tempos.append(r[i]['tempo'])
        lags.append(lag)

    # Create raw lag histogram
    int_lags = [int(round(x)) for x in lags]
    min_lag = 20
    max_lag = 120
    hainsworth_prior_histogram = np.zeros((int(max_lag+1-min_lag),))
    all_lags = list(range(min_lag, max_lag+1))
    for i in int_lags:
        hainsworth_prior_histogram[i-min_lag] += 1
    #plt.bar(all_lags, hainsworth_prior_histogram)

    def add_k_smoothing(histo, k=1):
        newhisto = []
        for i in range(len(histo)):
            newhisto.append(histo[i] + k)
        return newhisto

    def gaussian_smoothing(histo):
        standard_dev = np.std(histo)
        lags = len(histo)
        ghisto = gaussian(20, standard_dev/2)
        ghisto /= ghisto.sum()
        #print(max(ghisto))
        #print(ghisto)
        a = filtfilt(ghisto, [1], histo)
        return a

    # Smooth the histogram
    h1 = add_k_smoothing(hainsworth_prior_histogram, k)
    h2 = gaussian_smoothing(hainsworth_prior_histogram)
    h3 = gaussian_smoothing(h1)

    # Normalize and plot
    h1 /= sum(h1)
    h2 /= sum(h2)
    h3 /= sum(h3)
    if show:
        plt.bar(all_lags, h1, width=1)
        plt.bar(all_lags, h2, width=1)
        plt.bar(all_lags, h3, width=1)

    return h1, h2, h3
