# -*- coding: utf-8 -*-
"""
author: Miroslav Jirik
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np
import filtration


def noises(shape, voxelsize=None, exponent=0, lambda_start=0, lambda_range=1):
    """
    generate noise based on space properties
    :return:
    """
    if voxelsize is None:
        voxelsize = np.ones([1, len(shape)])

    freq_start = 1.0/lambda_start
    freq_range = 1.0/lambda_range


def noisef(shape, fs=None, return_spectrum=False, random_generator_seed=None, exponent=0, freq_start=0, freq_range=-1, spectrum=None):
    """
    Generate noise based on FFT transformation. Complex ndarray is generated as a seed for fourier spectre.
    The specter is filtered based on power function of frequency. This is controled by exponent parameter.
    Then lowpass and hipass filter are applied.

    :param shape: size of output data
    :param return_spectrum:
    :param random_generator_seed:

    For other parameters see process_specturum_seed().
    :return:
    """
    if fs is None:
        fs = np.ones(len(shape))
        #fs = np.ones([1, len(shape)])


    if random_generator_seed is not None:
        np.random.seed(seed=random_generator_seed)

    if spectrum is None:
        spectrum = generate_spectrum_seed(shape)

    signal, filter, spectrum = spectrum_filtration(
        spectrum,
        fs=fs,
        exponent=exponent,
        freq_start=freq_start,
        freq_range=freq_range
    )

    if return_spectrum:
        return signal, filter, spectrum
    return signal


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

    dist = filtration.dist_from_center(spectrum.shape, axis_size=voxelsize)

    shspectrum = np.fft.fftshift(spectrum)
    pfilter = filtration.power_filter(shspectrum.shape, exponent, dist=dist)
    shspectrum = filtration.apply_filter(shspectrum, pfilter)

    filt = filtration.hipass_fft_mask(spectrum.shape, freq_start, dist=dist)
    if freq_range is not None:
        filt *= filtration.lopass_fft_mask(spectrum.shape, freq_start + freq_range, dist=dist)
    shspectrum *= filt
    spectrum = np.fft.ifftshift(shspectrum)

    signal = np.real(np.fft.ifftn(spectrum))
    return signal, filter, spectrum


def generate_spectrum_seed(shape, seed=None):
    im = (np.random.random(shape) * 2.0) - 1.0
    re = (np.random.random(shape) * 2.0) - 1.0
    spectrum = (re + 1j * im) / 2**0.5
    return spectrum

def show(real_signal, filter_shifted=None, spectrum=None, log_view=False):
    import matplotlib.pyplot as plt

    plt.gray()
    plt.subplot(231)
    plt.imshow(real_signal)
    plt.colorbar()

    # shfilter = np.fft.fftshift(filter)
    plt.subplot(232)
    plt.imshow(np.abs(filter_shifted))
    plt.colorbar()

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

