# -*- coding: utf-8 -*-
"""
author: Miroslav Jirik
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np

import scipy
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

def power_filter(shape, e, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = np.power(dist, e)
    return output

def lopass_fft_mask(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist <= radius
    return output

def hipass_fft_mask(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist > radius
    return output

# TODO voxelsize dependency
def dist_from_center(shape, axis_size=None):
    if axis_size is None:
        axis_size = np.ones(len(shape))
    axis_size = np.asarray(axis_size, dtype=np.float)

    center = (np.asarray(shape) - 1) / 2.0


    xi = []
    for i in range(len(shape)):
        xi.append(range(shape[i]))
    yi = np.meshgrid(*xi, indexing='ij')

    dist = np.zeros(shape)
    for i in range(len(shape)):
        dist += ((yi[i] - center[i]) * axis_size[i]) ** 2
    dist = dist**0.5
    return dist

def fftfreq(shape, spacing=None):
    """
    N dimensional variant of numpy.fft.fftfreq

    :param shape:
    :param spacing: if None it is set to 1. If it is scalar it is used for every axis
    :return:
    """
    input = np.zeros(shape, dtype=np.bool)

    # set first pixel (for general dimension) to one
    # input[np.zeros([len(input.shape), 1], dtype=int).tolist()] = 1
    # shinput = np.fft.fftshift(input)
    # shdist = scipy.ndimage.morphology.distance_transform_edt(shinput, sampling=spacing)
    # dist = np.fft.ifftshift(shdist)

    if np.isscalar(shape):
        shape = [shape]

    if spacing is None:
        spacing = np.ones(len(shape))

    if np.isscalar(spacing):
        spacing = spacing * np.ones(len(shape))

    spacing = np.asarray(spacing, dtype=np.float)

    center = np.ceil((np.asarray(shape) - 1) / 2.0)

    xi = []
    for i in range(len(shape)):
        xi.append(range(shape[i]))
    yi = np.meshgrid(*xi, indexing='ij')

    shdist = np.zeros(shape)
    for i in range(len(shape)):
        shdist += (((yi[i] - center[i]) / (spacing[i] * shape[i]) ) ** 2) # * spacing[i] / shape[i]
    shdist = (shdist**0.5)

    dist = np.fft.ifftshift(shdist)
    return dist





def P2R(radii, angles):
    radii = np.asarray(radii)
    angles = np.asarray(angles)
    return radii * np.exp(1j*angles)

def R2P(x):
    return np.abs(x), np.angle(x)

def spectrum_filtration(spectrum, fs=None, exponent=0, freq_start=0, freq_range=None):
    """
    Filter spectrum based on frequency
    :param spectrum:
    :param exponent:
    :param freq_start:
    :param freq_range:
    :return:
    """
    if freq_range< 0:
        freq_range = None

    if fs is None:
        fs = np.ones(len(spectrum.shape))

    voxelsize = 1.0 / np.asarray(fs)

    dist = dist_from_center(spectrum.shape, axis_size=voxelsize)
    dist = fftfreq(spectrum.shape, spacing=voxelsize)

    shspectrum = np.fft.fftshift(spectrum)
    pfilter = power_filter(shspectrum.shape, exponent, dist=dist)
    shspectrum = apply_filter(shspectrum, pfilter)

    filt = hipass_fft_mask(spectrum.shape, freq_start, dist=dist)
    if freq_range is not None:
        filt *= lopass_fft_mask(spectrum.shape, freq_start + freq_range, dist=dist)
    shspectrum *= filt
    spectrum = np.fft.ifftshift(shspectrum)

    signal = np.real(np.fft.ifftn(spectrum))
    return signal, filter, spectrum
