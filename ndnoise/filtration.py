# -*- coding: utf-8 -*-
"""
author: Miroslav Jirik
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np

import scipy
# from scipy.signal import butter, lfilter, freqz
import scipy.signal
import matplotlib.pyplot as plt


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.lfilter(b, a, data)
    return y

def butter_bandpass_freq_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    dataf = np.fft.fft(data)
    # w, h = freqz(b, a, worN=len(dataf), whole=True)
    w, h = scipy.signal.freqz(b, a, worN=len(dataf), whole=True)
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

def lopass_ideal_fft_mask(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist <= radius
    return output

def hipass_ideal_fft_mask(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist > radius
    return output

def lopass_butter_fft_mask(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist <= radius
    return output

def hipass_butter_fft_mask(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist > radius
    return output

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

def spectrum_filtration(spectrum, fs=None, exponent=0, freq_start=0, freq_range=None, filter_type="butter"):
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

    if filter_type is "ideal":
        lopass_fcn = lopass_ideal_fft_mask
        hipass_fcn = hipass_ideal_fft_mask
    elif filter_type is "butter":
        lopass_fcn = lopass_butter_fft_mask
        hipass_fcn = hipass_butter_fft_mask
    else:
        logger.error("Unknown filter type")

    # dist = dist_from_center(spectrum.shape, axis_size=voxelsize)
    freqs = fftfreq(spectrum.shape, spacing=voxelsize)

    # shspectrum = np.fft.fftshift(spectrum)
    filt = power_filter(spectrum.shape, exponent, dist=freqs)
    # spectrum = apply_filter(spectrum, pfilter)

    filt *= hipass_fcn(spectrum.shape, freq_start, dist=freqs)
    if freq_range is not None:
        filt *= lopass_fcn(spectrum.shape, freq_start + freq_range, dist=freqs)
    spectrum = apply_filter(spectrum, filt)
    # spectrum *= filt
    # spectrum = np.fft.ifftshift(shspectrum)

    signal = np.real(np.fft.ifftn(spectrum))
    return signal, filt, spectrum, freqs

def show(real_signal, filter=None, spectrum=None, log_view=False):
    import matplotlib.pyplot as plt

    # plt.gray()
    ax = plt.subplot(231)
    plt.imshow(real_signal, cmap="gray")
    plt.colorbar()
    ax.set_title("signal")

    # shfilter = np.fft.fftshift(filter)
    ax = plt.subplot(232)
    plt.imshow(np.abs(filter))
    plt.colorbar()
    ax.set_title("filter")

    shspecturm = np.fft.fftshift(spectrum)
    if log_view:
        shspecturm = np.log(shspecturm)


    ax = plt.subplot(233)
    ax.imshow(np.abs(shspecturm))
    plt.colorbar()
    ax.set_title("abs")

    ax = plt.subplot(234)
    ax.imshow(np.angle(shspecturm))
    plt.colorbar()
    ax.set_title("angle")

    ax = plt.subplot(235)
    ax.imshow(np.real(shspecturm))
    plt.colorbar()
    ax.set_title("real")

    ax = plt.subplot(236)
    ax.imshow(np.imag(shspecturm))
    plt.colorbar()
    ax.set_title("imag")


    # plt.show()


