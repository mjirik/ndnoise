# -*- coding: utf-8 -*-
"""
author: Miroslav Jirik
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np
import filtration


def noises(shape, sample_spacing=None, exponent=0, lambda_start=0, lambda_range=1, **kwargs):
    """
    generate noise based on space properties
    :return:
    """
    if sample_spacing is None:
        sample_spacing = np.ones([1, len(shape)])

    sample_spacing = np.asarray(sample_spacing)
    freq_start = 1.0 / lambda_start
    freq_range = 1.0 / lambda_range
    sampling_frequency = 1.0 / sample_spacing
    retval = noisef(
        shape,
        sampling_frequency=sampling_frequency,
        exponent=exponent,
        freq_start=freq_start,
        freq_range=freq_range,
        **kwargs
    )
    return retval


def noisef(shape, sampling_frequency=None, return_spectrum=False, random_generator_seed=None, exponent=0, freq_start=0, freq_range=-1, spectrum=None):
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
    if sampling_frequency is None:
        sampling_frequency = np.ones(len(shape))
        #fs = np.ones([1, len(shape)])


    if random_generator_seed is not None:
        np.random.seed(seed=random_generator_seed)

    if spectrum is None:
        spectrum = generate_spectrum_seed(shape)

    signal, filter, spectrum = filtration.spectrum_filtration(
        spectrum,
        fs=sampling_frequency,
        exponent=exponent,
        freq_start=freq_start,
        freq_range=freq_range
    )

    if return_spectrum:
        return signal, filter, spectrum
    return signal




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

