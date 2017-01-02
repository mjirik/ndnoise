# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)

import unittest
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import numpy as np
import ndnoise.filtration
import ndnoise.generator


class MyTestCase(unittest.TestCase):



    def test_butter(self):

        # Sample rate and desired cutoff frequencies (in Hz).
        fs = 5000.0
        lowcut = 500.0
        highcut = 1250.0

        # Plot the frequency response for a few different orders.
        plt.figure(1)
        plt.clf()
        for order in [3, 6, 9]:
            b, a = ndnoise.filtration.butter_bandpass(lowcut, highcut, fs, order=order)
            w, h = freqz(b, a, worN=2000)
            # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)


        # Filter a noisy signal.
        T = 0.05
        nsamples = T * fs
        t = np.linspace(0, T, nsamples, endpoint=False)
        a = 0.02
        f0 = 600.0

        ys = a * np.cos(2 * np.pi * f0 * t + .11)
        x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
        x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
        x += a * np.cos(2 * np.pi * f0 * t + .11)
        x += 0.03 * np.cos(2 * np.pi * 2000 * t)

        y1 = ndnoise.filtration.butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
        y2 = ndnoise.filtration.butter_bandpass_freq_filter(x, lowcut, highcut, fs, order=6)

        energy_ys = np.sum(ys**2)
        energy_error_y1 = np.sum((ys - y1)**2)
        energy_error_y2 = np.sum((ys - y2)**2)
        energy_error_y2_y1 = np.sum((y1 - y2)**2)

        self.assertLess(energy_error_y2_y1, energy_ys * 0.2, "Both filtered signals are similar")
        # self.assertLess(energy_error_y1, energy_ys * 0.1)

        # plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
        #          '--', label='sqrt(0.5)')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Gain')
        # plt.grid(True)
        # plt.legend(loc='best')
        #
        # plt.figure(2)
        # plt.clf()
        # plt.plot(t, x, label='Noisy signal')
        #
        # plt.plot(t, ys, label='Original signal (%g Hz)' % f0)
        # plt.plot(t, y1, label='Filtered signal (%g Hz)' % f0)
        # plt.plot(t, y2 + 0.0, label='Filtered signal FFT (%g Hz)' % f0)
        # plt.xlabel('time (seconds)')
        # plt.hlines([-a, a], 0, T, linestyles='--')
        # plt.grid(True)
        # plt.axis('tight')
        # plt.legend(loc='upper left')
        #
        # plt.show()

    def test_dist(self):
        import ndnoise
        dst = ndnoise.filtration.dist_from_center([5, 5])
        self.assertAlmostEqual(dst[0][0], 2 * np.sqrt(2))

    def test_dist_with_defined_voxelsize(self):
        import ndnoise
        dst = ndnoise.filtration.dist_from_center([5, 5], [2.0, 2.0])
        self.assertAlmostEqual(dst[2][2], 0)
        self.assertAlmostEqual(dst[0][0], 4 * np.sqrt(2))

    def test_dist_with_defined_voxelsize_3d(self):
        import ndnoise
        dst = ndnoise.filtration.dist_from_center([5, 5, 5], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(dst[2][2][2], 0)
        self.assertAlmostEqual(dst[0][0][0], np.sqrt(2**2 + 4**2 + 6**2))

    def test_max_x_freq_spectrum_processing(self):
        """
        Test of maximal frequence
        :return:
        """
        shape = [24, 24]
        center = (np.asarray(shape) / 2)

        shspectrum = np.zeros(shape=shape, dtype=np.complex)
        shspectrum[center[0], 0] = 1
        # shspectrum[center[0], 24] = 0
        shspectrum[center[0], center[1]] = 2
        spectrum = np.fft.ifftshift(shspectrum)

        signal, filt, spectrum_ret = ndnoise.generator.noisef(shape, spectrum=spectrum, return_spectrum=True)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(signal, cmap="gray")
        # plt.subplot(132)
        # plt.imshow(np.abs(shspectrum), cmap="gray")
        # plt.subplot(133)
        # plt.imshow(np.abs(spectrum_ret), cmap="gray")
        # plt.show()

        # a = signal[[:]]
        suda = signal[:, ::2]
        licha = signal[:, 1::2]

        var0 = np.var(signal)
        vars = np.var(suda)
        varl = np.var(licha)

        self.assertGreater(var0, 2.5 * vars)
        self.assertGreater(var0, 2.5 * varl)


    def test_max_y_freq_spectrum_processing(self):
        """
        Test of maximal frequence
        :return:
        """
        shape = [20, 20]
        # center = np.round((np.asarray(shape) / 2.0))
        center = (np.asarray(shape) / 2)

        shspectrum = np.zeros(shape=shape, dtype=np.complex)
        shspectrum[0, center[1]] = 1
        # shspectrum[center[0], 24] = 0
        shspectrum[center[0], center[1]] = 2
        spectrum = np.fft.ifftshift(shspectrum)

        signal, filt, spectrum_ret = ndnoise.generator.noisef(shape, spectrum=spectrum, return_spectrum=True)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.subplot(131)
        # plt.imshow(signal, cmap="gray")
        # plt.subplot(132)
        # plt.imshow(np.abs(shspectrum), cmap="gray")
        # plt.subplot(133)
        # plt.imshow(np.abs(spectrum_ret), cmap="gray")
        # plt.show()

        # a = signal[[:]]
        suda = signal[::2, :]
        licha = signal[1::2, :]

        var0 = np.var(signal)
        vars = np.var(suda)
        varl = np.var(licha)

        self.assertGreater(var0, 2.5 * vars)
        self.assertGreater(var0, 2.5 * varl)

    def test_compare_fftfreq(self):
        """
        Test few setups
        :return:
        """
        import itertools
        sampling_rates = [10, 15, 2.1, 0.01]
        ns = [4, 9, 5, 6]
        for params in itertools.product(sampling_rates, ns):
            sampling_rate, n = params
            freq1 = np.fft.fftfreq(n, d=1./sampling_rate)
            freq2 = ndnoise.filtration.fftfreq([n], [1. / sampling_rate])
            logger.debug("freq1 " + str(freq1))
            logger.debug("freq2 " + str(freq2))

            self.assertAlmostEqual(0.0, np.sum((np.abs(freq1) - freq2)**2), msg="parameters " + str(params))

    def test_2d_fftfreq(self):
        import itertools
        spacings = [[10, 1.0], [0.12, 5.0], None, 3]
        shapes = [[10, 11], [9, 11], [8, 10], [15, 2]]
        for params in itertools.product(spacings, shapes):
            spacing, shape = params
            freq = ndnoise.filtration.fftfreq(shape, spacing=spacing)
            logger.debug("freq " + str(freq))

    def test_nd_fftfreq(self):
        import itertools
        spacings = [None, 5, 1.2]
        shapes = [[2], 3, [10, 11], [9, 11, 15], [8, 10, 5, 2], [15, 2, 3, 2, 5]]
        for params in itertools.product(spacings, shapes):
            spacing, shape = params
            freq = ndnoise.filtration.fftfreq(shape, spacing=spacing)
            logger.debug("freq " + str(freq))


if __name__ == '__main__':
    unittest.main()
