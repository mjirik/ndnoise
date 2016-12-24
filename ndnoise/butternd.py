# -*- coding: utf-8 -*-
"""
author: Miroslav Jirik
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np

from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass_freq_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    dataf = np.fft.fft(data)
    # w, h = freqz(b, a, worN=len(dataf), whole=True)
    w, h = freqz(b, a, worN=len(dataf), whole=True)
    plt.figure(3)
    plt.subplot(211)
    nyq = (fs * 0.5 / np.pi)

    plt.plot(w*nyq, abs(h))
    # filtered_spectrum = dataf * np.real(h)
    # filtered_spectrum = dataf * np.abs(h)
    filtered_spectrum = dataf * h
    filtered = np.fft.ifft(filtered_spectrum)
    # plt.subplot(212)
    # plt.plot(abs(dataf), label='data')
    # plt.plot(abs(h), label='filter')
    # plt.legend()
    # plt.show()
    return np.real(filtered)
    # return np.abs(filtered)
