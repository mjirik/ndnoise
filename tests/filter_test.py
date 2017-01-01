import unittest
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
import numpy as np
import ndnoise.filter


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
            b, a = ndnoise.filter.butter_bandpass(lowcut, highcut, fs, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)


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

        y1 = ndnoise.filter.butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
        y2 = ndnoise.filter.butter_bandpass_freq_filter(x, lowcut, highcut, fs, order=6)

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
        dst = ndnoise.filter.dist_from_center([5, 5])
        self.assertAlmostEqual(dst[0][0], 2 * np.sqrt(2))

    def test_dist_with_defined_voxelsize(self):
        import ndnoise
        dst = ndnoise.filter.dist_from_center([5, 5], [2.0, 2.0])
        self.assertAlmostEqual(dst[2][2], 0)
        self.assertAlmostEqual(dst[0][0], 4 * np.sqrt(2))

    def test_dist_with_defined_voxelsize_3d(self):
        import ndnoise
        dst = ndnoise.filter.dist_from_center([5, 5, 5], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(dst[2][2][2], 0)
        self.assertAlmostEqual(dst[0][0][0], np.sqrt(2**2 + 4**2 + 6**2))

if __name__ == '__main__':
    unittest.main()
