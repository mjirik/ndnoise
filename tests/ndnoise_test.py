import unittest

# import sed3
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

import ndnoise

class MyTestCase(unittest.TestCase):
    def test_something(self):
        noise, filter, spectrum, freqs = ndnoise.noisef([100, 100], return_spectrum=True)


        # ndnoise.show(noise, filter, spectrum)
        # self.assertEqual(True, False)

    @unittest.skip("interactive")
    def test_butter(self):
        b, a = scipy.signal.butter(4, [0, 10000], btype='band', analog=True)
        w, h = scipy.signal.freqs(b, a, worN=range(1000, 2000))
        plt.plot(w, abs(h))
        # plt.show()

    def test_voxelsize(self):
        import scipy.misc
        lena = scipy.misc.ascent()
        spectrum = np.fft.fftn(lena)

        spectrum = ndnoise.generator.generate_spectrum_seed([100, 100])

        out = ndnoise.filtration.spectrum_filtration(
            spectrum,
            fs=[0.1, 0.5],
            freq0=0,
            freq1=10,
            exponent=-1.5
        )
        signal, filter, spectrum, freqs = out

        # ndnoise.show(signal, filter, spectrum, log_view=True)

    # lambda_mm = shape_px*(1/freq_px)*voxelsize_mm
    # lambda_mm * freq_px = shape_px*voxelsize_mm
    def test_3d_noise(self):
        import scipy.misc
        lena = scipy.misc.ascent()
        spectrum = np.fft.fftn(lena)

        noise, filt, spectrum, freqs = ndnoise.noisef(
            [100,102,103],
            random_generator_seed=5,
            freq0=0.1,
            freq1=2 / 10.0,
            exponent=1.0,
            return_spectrum=True
        )
        # plt.subplot(221)
        # plt.imshow(noise[5,:,:], cmap="gray")
        # plt.colorbar()
        # plt.subplot(222)
        # plt.imshow(np.abs(filt[5,:,:]))
        # plt.colorbar()
        # plt.subplot(223)
        # plt.imshow(np.abs(spectrum[5,:,:]))
        # plt.colorbar()
        # plt.subplot(224)
        # plt.imshow(freqs[5,:,:])
        # plt.colorbar()
        # plt.show()

        # self.assertEqual(True, False)

    def test_3d_noises(self):
        import scipy.misc
        lena = scipy.misc.ascent()
        spectrum = np.fft.fftn(lena)

        noise = ndnoise.generator.noises(
            [100,102,103],
            sample_spacing=[1,1,1],
            random_generator_seed=5,
            lambda0=20,
            lambda1=22,
            exponent=0.0
        )
        return noise


    @unittest.skip("interactive")
    def test_3d_noises_interactive(self):
        noise = self.test_3d_noises()
        plt.figure()
        plt.subplot(121)
        plt.imshow(noise[5,:,:], cmap="gray")
        plt.subplot(122)
        plt.imshow(np.abs((np.fft.fft(noise)))[0,:,:])
        # plt.show()


    #def test_3d_noises_profile(self):
    #
    #    import profile
    #    def fcn():
    #        return self.test_3d_noises()
    #
    #    profile.run("print dir()")
    #    self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
