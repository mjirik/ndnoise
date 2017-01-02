import unittest

import sed3
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import ndnoise

class MyTestCase(unittest.TestCase):
    def test_something(self):
        noise, filter, spectrum = ndnoise.noisef([100, 100], return_spectrum=True)


        ndnoise.show(noise, filter, spectrum)
        # self.assertEqual(True, False)

    def test_butter(self):
        b, a = scipy.signal.butter(4, [0, 10000], btype='band', analog=True)
        w, h = scipy.signal.freqs(b, a)
        plt.plot(w, abs(h))
        plt.show()


    def test_voxelsize(self):
        import scipy.misc
        lena = scipy.misc.ascent()
        spectrum = np.fft.fftn(lena)

        spectrum = ndnoise.generator.generate_spectrum_seed([100, 100])

        out = ndnoise.generator.spectrum_filtration(
            spectrum,
            voxelsize=[1, 2],
            freq_start=0,
            freq_range=10,
            exponent=-1.5
        )
        signal, filter, spectrum = out

        ndnoise.show(signal, filter, spectrum, log_view=True)


    def test_lena_filter(self):
        import scipy.misc
        lena = scipy.misc.ascent()
        spectrum = np.fft.fftn(lena)

        spectrum = ndnoise.generator.generate_spectrum_seed([100, 100])

        out = ndnoise.generator.spectrum_filtration(
            spectrum,
            freq_start=0,
            freq_range=10,
            exponent=-1.5
        )
        signal, filter, spectrum = out

        ndnoise.show(signal, filter, spectrum, log_view=True)
        # self.assertEqual(True, False)

    # lambda_mm = shape_px*(1/freq_px)*voxelsize_mm
    # lambda_mm * freq_px = shape_px*voxelsize_mm
    def test_3d_noise(self):
        import scipy.misc
        lena = scipy.misc.ascent()
        spectrum = np.fft.fftn(lena)

        noise = ndnoise.noisef(
            [100,102,103],
            random_generator_seed=5,
            lambda_start=0,
            lambda_range=1/10.0,
            exponent=-1.5
        )

        plt.imshow(noise[5,:,:])
        # plt.show()

        # self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()
