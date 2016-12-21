import unittest

import sed3
import matplotlib.pyplot as plt
import numpy as np

import ndnoise

class MyTestCase(unittest.TestCase):
    def test_something(self):
        noise, filter, spectrum = ndnoise.generate([100, 100], return_spectrum=True)


        ndnoise.show(noise, filter, spectrum)
        # self.assertEqual(True, False)

    def test_lena_filter(self):
        import scipy.misc
        lena = scipy.misc.ascent()
        spectrum = np.fft.fftn(lena)

        spectrum = ndnoise.generator.generate_spectrum_seed([100,100])

        out = ndnoise.generator.process_spectrum_seed(
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

        noise = ndnoise.generate(
            [100,102,103],
            random_generator_seed=5,
            freq_start=0,
            freq_range=10,
            exponent=-1.5
        )

        plt.imshow(noise[5,:,:])
        plt.show()

        # self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()
