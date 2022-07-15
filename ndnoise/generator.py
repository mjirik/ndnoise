# -*- coding: utf-8 -*-
"""
author: Miroslav Jirik
"""

import logging
logger = logging.getLogger(__name__)
import numpy as np
import scipy
from . import filtration

def noises(shape, sample_spacing=None, exponent=0, lambda0=0, lambda1=1, method="space", **kwargs):
    """ Create noise based on space paramters.

    :param shape:
    :param sample_spacing: in space units like milimeters
    :param exponent:
    :param lambda0: wavelength of first noise
    :param lambda1: wavelength of last noise
    :param method: use "space" or "freq" method. "freq" is more precise but slower.
    :param kwargs:
    :return:
    """
    kwargs1 = dict(
        shape=shape,
        sample_spacing=sample_spacing,
        exponent=exponent,
        lambda0=lambda0,
        lambda1=lambda1,
        **kwargs
    )

    if method is "space":
        noise = noises_space(**kwargs1)
    elif method is "freq":
        noise = noises_freq(**kwargs1)
    else:
        logger.error("Unknown noise method `{}`".format(method))

    return noise

def ndimage_normalization(data, std_factor=1.0):
    t0 = datetime.datetime.now()
    data0n = (data - np.mean(data)) * 1.0 / (std_factor * np.var(data)**0.5)
    logger.debug(f"t_norm={datetime.datetime.now() - t0}")
    
    return data0n

def gaussian_filter_fft(image, sigma):
    input_ = np.fft.fftn(image)
    result = scipy.ndimage.fourier_gaussian(input_, sigma=sigma)
    result = np.fft.ifftn(result)
    return np.abs(result)

def noises_space(
        shape,
        sample_spacing=None,
        exponent=0.0,
        lambda0=0,
        lambda1=1,
        random_generator_seed=None,
        use_fft="auto",
        **kwargs
):
    """
    use_fft: ("auto", "lambda0", "lambda1", "both", "none") auto: use fft if lambda is > 5
    """

    data0 = 0
    data1 = 0
    w0 = 0
    w1 = 0
    lambda0_px = None
    lambda1_px = None
    if random_generator_seed is not None:
        np.random.seed(seed=random_generator_seed)
        
    use_fft_l0 = use_fft == "lambda0" or use_fft == "both"
    use_fft_l1 = use_fft == "lambda1" or use_fft == "both"
    if use_fft == "auto":
        use_fft_l0 = lambda0 > 5
        use_fft_l1 = lambda1 > 5

    # lambda1 = lambda_stop * np.asarray(sample_spacing)
    t0 = datetime.datetime.now()

    if lambda0 is not None:
        lambda0_px = lambda0 / np.asarray(sample_spacing)
        data0 = np.random.rand(*shape)
        if use_fft_l0:
            data0 = gaussian_filter_fft(data0, sigma=lambda1_px)
            pass
        else:
            data0 = scipy.ndimage.filters.gaussian_filter(data0, sigma=lambda0_px)
        data0 = ndimage_normalization(data0)
        w0 = np.exp(exponent * lambda0)
    logger.debug(f"t_l0={datetime.datetime.now() - t0}")
    t0 = datetime.datetime.now()
    if lambda1 is not None:
        lambda1_px = lambda1 / np.asarray(sample_spacing)
        data1 = np.random.rand(*shape)
        if use_fft_l1:
            data1 = gaussian_filter_fft(data1, sigma=lambda1_px)
        else:
            data1 = scipy.ndimage.filters.gaussian_filter(data1, sigma=lambda1_px)

        data1 = ndimage_normalization(data1)
        w1 = np.exp(exponent * lambda1)
    logger.debug(f"t_l1={datetime.datetime.now() - t0}")
    t0 = datetime.datetime.now()
    logger.debug("lambda_px {} {}".format(lambda0_px, lambda1_px))
    logger.debug(f"use_fft lambda 0 and 1 {use_fft_l0} {use_fft_l1}")
    wsum = w0 + w1
    if wsum > 0:
        w0 = w0 / wsum
        w1 = w1 / wsum

    # print w0, w1
    # print np.mean(data0), np.var(data0)
    # print np.mean(data1), np.var(data1)

    data = ( data0 * w0 +  data1 * w1)
    logger.debug("w0, w1 {} {}".format(w0, w1))

    # plt.figure()
    # plt.imshow(data0[:,:,50], cmap="gray")
    # plt.colorbar()
    # plt.figure()
    # plt.imshow(data1[:,:,50], cmap="gray")
    # plt.colorbar()
    return data


