# -*- coding: utf-8 -*-
"""
author: Miroslav Jirik
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np
import scipy
import filtration


def noises(shape, sample_spacing=None, exponent=0, lambda_start=0, lambda_stop=1, method="space", **kwargs):
    """ Create noise based on space paramters.

    :param shape:
    :param sample_spacing: in space units like milimeters
    :param exponent:
    :param lambda_start: wavelength of first noise
    :param lambda_stop: wavelength of last noise
    :param method: use "space" or "freq" method. "freq" is more precise but slower.
    :param kwargs:
    :return:
    """
    kwargs1 = dict(
        shape=shape,
        sample_spacing=sample_spacing,
        exponent=exponent,
        lambda_start=lambda_start,
        lambda_stop=lambda_stop,
        **kwargs
    )

    if method is "space":
        noise = noises_space(**kwargs1)
    elif method is "freq":
        noise = noises_freq(**kwargs1)
    else:
        logger.error("Unknown noise method `{}`".format(method))

    return noise

def noise_normalization(data, std_factor=1.0):
    data0n = (data - np.mean(data)) * 1.0 / (std_factor * np.var(data)**0.5)
    return data0n


def noises_space(
        shape,
        sample_spacing=None,
        exponent=0.0,
        lambda_start=0,
        lambda_stop=1,
        **kwargs
):

    data0 = 0
    data1 = 0
    w0 = 0
    w1 = 0

    lambda1 = lambda_stop * np.asarray(sample_spacing)

    if lambda_start is not None:
        lambda0 = lambda_start * np.asarray(sample_spacing)
        data0 = np.random.rand(*shape)
        data0 = scipy.ndimage.filters.gaussian_filter(data0, sigma=lambda0)
        data0 = noise_normalization(data0)
        w0 = np.exp(exponent * lambda_start)

    if lambda_stop is not None:
        lambda1 = lambda_stop * np.asarray(sample_spacing)
        data1 = np.random.rand(*shape)
        data1 = scipy.ndimage.filters.gaussian_filter(data1, sigma=lambda1)
        data1 = noise_normalization(data1)
        w1 = np.exp(exponent * lambda_stop)

    wsum = w0 + w1
    if wsum > 0:
        w0 = w0 / wsum
        w1 = w1 / wsum

    print w0, w1
    print np.mean(data0), np.var(data0)
    print np.mean(data1), np.var(data1)

    data = ( data0 * w0 +  data1 * w1)

    # plt.figure()
    # plt.imshow(data0[:,:,50], cmap="gray")
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(data1[:,:,50], cmap="gray")
    # plt.colorbar()
    return data

def noises_freq(shape, sample_spacing=None, exponent=0, lambda_start=0, lambda_stop=1, **kwargs):
    """Generate noise based on space properties using fft transforamtion.
    :return:
    """
    if sample_spacing is None:
        sample_spacing = np.ones([1, len(shape)])
    sample_spacing = np.asarray(sample_spacing)
    sampling_frequency = 1.0 / sample_spacing

    if lambda_start is None or lambda_start == 0:
        freq_stop = None
    else:
        freq_stop = 1.0 / lambda_start

    if lambda_stop is None or lambda_stop == 0:
        freq_start = None
    else:
        freq_start = 1.0 / lambda_stop

    retval = noisef(
        shape,
        sampling_frequency=sampling_frequency,
        exponent=exponent,
        freq_start=freq_start,
        freq_stop=freq_stop,
        **kwargs
    )
    return retval


def noisef(shape, sampling_frequency=None, return_spectrum=False, random_generator_seed=None, exponent=0, freq_start=0, freq_stop=-1, spectrum=None):
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

    signal, filter, spectrum, freqs = filtration.spectrum_filtration(
        spectrum,
        fs=sampling_frequency,
        exponent=exponent,
        freq_start=freq_start,
        freq_stop=freq_stop
    )

    if return_spectrum:
        return signal, filter, spectrum, freqs
    return signal




def generate_spectrum_seed(shape, seed=None):
    im = (np.random.random(shape) * 2.0) - 1.0
    re = (np.random.random(shape) * 2.0) - 1.0
    spectrum = (re + 1j * im) / 2**0.5
    return spectrum

