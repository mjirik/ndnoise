# -*- coding: utf-8 -*-
"""
author: Miroslav Jirik
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np

def generate(shape, voxelsize=None, return_spectrum=False, random_generator_seed=None, exponent=0, lambda_start=0, lambda_range=-1):
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
    if voxelsize is None:
        voxelsize = np.ones([1, len(shape)])

    freq_start = 1.0/lambda_start
    freq_range = 1.0/lambda_range

    if random_generator_seed is not None:
        np.random.seed(seed=random_generator_seed)
    spectrum = generate_spectrum_seed(shape)
    signal, filter, spectrum = process_spectrum_seed(
        spectrum,
        voxelsize=voxelsize,
        exponent=exponent,
        freq_start=freq_start,
        freq_range=freq_range
    )

    if return_spectrum:
        return signal, filter, spectrum
    return signal

def process_spectrum_seed(spectrum, voxelsize=None, exponent=0, freq_start=0, freq_range=None):
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

    if voxelsize is None:
        voxelsize = np.ones([1, len(spectrum.shape)])

    dist = construct_filter_dist(spectrum.shape, voxelsize=voxelsize)

    shspectrum = np.fft.fftshift(spectrum)
    pfilter = power_filter(shspectrum.shape, exponent)
    shspectrum = apply_filter(shspectrum, pfilter)

    filter = hipass_filter(spectrum.shape, freq_start, dist=dist)
    if freq_range is not None:
        filter *= lopass_filter(spectrum.shape, freq_start + freq_range, dist=dist)
    shspectrum *= filter
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

def lopass_filter(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist < radius
    return output

def hipass_filter(shape, radius, dist=None):
    # if dist is None:
    #     dist = construct_filter_dist(shape)
    output = dist > radius
    return output

# TODO voxelsize dependency
def construct_filter_dist(shape, voxelsize=None):
    voxelsize = np.asarray(voxelsize, dtype=np.float)
    if voxelsize is None:
        voxelsize = np.ones(len(shape))


    center = (np.asarray(shape)/voxelsize - 1) / 2.0


    xi = []
    for i in range(len(shape)):
        xi.append(range(shape[i])/voxelsize[i])
    yi = np.meshgrid(*xi, indexing='ij')

    dist = np.zeros(shape)
    for i in range(len(shape)):
        dist += (yi[i] - center[i])**2
    dist = dist**0.5
    return dist

def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return np.abs(x), np.angle(x)
