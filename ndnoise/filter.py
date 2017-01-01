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


def apply_filter_on_abs(shspectrum, filter):
    radii, angle = R2P(shspectrum)
    radii *= filter
    shspectrum = P2R(radii, angle)
    return shspectrum

def apply_filter(shspectrum, filter):
    ab = np.real(shspectrum)
    im = np.imag(shspectrum)
    shspectrum = ab * filter + 1j * im * filter
    # radii, angle = R2P(shspectrum)
    # radii *= filter
    # shspectrum = P2R(radii, angle)
    return shspectrum

def apply_filter_on_abs(shspectrum, filter):
    radii, angle = R2P(shspectrum)
    radii *= filter
    shspectrum = P2R(radii, angle)
    return shspectrum


def power_filter(shape, e, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = np.power(dist, e)
    return output

def lopass_fft_mask(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist < radius
    return output

def hipass_fft_mask(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist > radius
    return output

# TODO voxelsize dependency
def dist_from_center(shape, voxelsize=None):
    if voxelsize is None:
        voxelsize = np.ones(len(shape))
    voxelsize = np.asarray(voxelsize, dtype=np.float)

    center = (np.asarray(shape) - 1) / 2.0


    xi = []
    for i in range(len(shape)):
        xi.append(range(shape[i]))
    yi = np.meshgrid(*xi, indexing='ij')

    dist = np.zeros(shape)
    for i in range(len(shape)):
        dist += ((yi[i] - center[i]) * voxelsize[i])**2
    dist = dist**0.5
    return dist

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return np.abs(x), np.angle(x)
