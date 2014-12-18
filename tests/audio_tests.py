import unittest
import numpy as np
from pymus.audio.waveform import Waveform
from pymus.audio.spectrogram import Spectrogram
from pymus.audio.effects import lowpass

class TestWaveform(unittest.TestCase):

    def setUp(self):
        # create a 5 Hz sine @ 1000 samples/sec
        self.f = 1000 # sampling frequency
        self.T = 1 / self.f # sampling period
        self.t = np.arange(0,1,self.T) # time axis
        self.samples = np.sin(2 * np.pi * 5.0 * self.t)
        # 100 samples / sec
        self.wf = Waveform(len(self.samples), self.f)
        self.wf[:] = self.samples;

    def test_init(self):
        self.assertEqual(len(self.t), len(self.wf))

    def test_iter(self):
        for i,v in enumerate(self.wf):
            self.assertEqual(self.samples[i], v._samples[0])

    def test_get(self):
        piece1 = self.wf[0:100]
        piece2 = self.wf[0:100:2]
        self.assertTrue(100, len(piece1))
        self.assertTrue(50, len(piece2))
        self.assertTrue(
            (piece1._samples==self.wf._samples[0:100]).all()
        )
        self.assertTrue(
            (piece2._samples==self.wf._samples[0:100:2]).all()
        )

    def test_set(self):
        self.assertFalse(self.wf._samples[2] == 1.0)
        self.wf[2] = 1.0
        self.assertTrue(self.wf._samples[2] == 1.0)
        self.wf[2] = 99999
        self.assertTrue(self.wf._samples[2] == 1.0)
        self.wf[2] = -99999
        self.assertTrue(self.wf._samples[2] == -1.0)

    def test_add(self):
        gained = self.wf + 0.5
        self.assertEqual(gained._samples[4], self.wf._samples[4] + 0.5)
        self.assertTrue(
            (gained._samples == self.wf._samples + 0.5).any()
        )
        self.assertTrue((gained._samples <= 1.0).all())
        self.assertTrue((gained._samples >= -1.0).all())

    def test_sub(self):
        gained = self.wf - 0.5
        self.assertEqual(gained._samples[2], self.wf._samples[2] - 0.5)
        self.assertTrue(
            (gained._samples == self.wf._samples - 0.5).any()
        )
        self.assertTrue((gained._samples <= 1.0).all())
        self.assertTrue((gained._samples >= -1.0).all())

    def test_mul(self):
        softened = self.wf * 0.8
        self.assertEqual(softened._samples[100], self.wf._samples[100] * 0.8)
        self.assertTrue(
            (softened._samples == self.wf._samples * 0.8).all()
        )
        self.assertTrue((softened._samples <= 1.0).all())
        self.assertTrue((softened._samples >= -1.0).all())

        loudened = self.wf * 1.2
        self.assertEqual(loudened._samples[10], self.wf._samples[10] * 1.2)
        self.assertTrue(
            (loudened._samples == self.wf._samples * 1.2).any()
        )
        self.assertTrue((loudened._samples <= 1.0).all())
        self.assertTrue((loudened._samples >= -1.0).all())

    def test_mean(self):
        self.assertAlmostEqual(self.wf.mean, np.sum(self.wf._samples)/len(self.wf))

    def test_std(self):
        m = self.wf.mean
        t = np.sqrt(np.sum([(x-m)**2 for x in self.wf._samples]) / len(self.wf))
        self.assertAlmostEqual(self.wf.std, t)

    def test_pow(self):
        powed = self.wf ** 2
        self.assertEqual(powed._samples[100], self.wf._samples[100] ** 2)
        self.assertTrue(
            (powed._samples == self.wf._samples ** 2).any()
        )
        self.assertTrue((powed._samples <= 1.0).all())
        self.assertTrue((powed._samples >= -1.0).all())

    def test_rms(self):
        added = (self.wf * 0.5) + 0.1
        self.assertAlmostEqual(self.wf.rms, np.std(self.wf._samples))
        # rms^2 = mean^2 + std^2
        self.assertAlmostEqual(added.rms**2, np.std(added._samples)**2 + np.mean(added._samples)**2)

    def test_resample(self):
        resampled = self.wf.resample(400)

        wf_bins = np.fft.fftfreq(len(self.wf), 1/self.wf.sample_rate)
        re_bins = np.fft.fftfreq(len(resampled), 1/resampled.sample_rate)

        wf_set = dict(zip(wf_bins, np.abs(self.wf.fft)))
        re_set = dict(zip(re_bins, np.abs(resampled.fft)))

        # The same frequencies should be positive in both fft's
        for freq,amp in wf_set.items():
            if freq in re_set and wf_set[freq] > 0:
                self.assertTrue(re_set[freq] > 0)
            elif freq in re_set:
                self.assertEqual(wf_set[freq], re_set[freq])

class TestSpectrogram(unittest.TestCase):

    def setUp(self):
        # create a funny waveform
        self.samples = (np.sin(np.linspace(0, 5 * 2 * np.pi, 500)) +  # 1 Hz
                        np.sin(np.linspace(0, 10 * 2 * np.pi, 500)) + # 2 Hz
                        np.sin(np.linspace(0, 50 * 2 * np.pi, 500))   # 10 Hz
                       )
        # 100 samples / sec
        self.wf = Waveform(len(self.samples), 100)
        self.wf[:] = self.samples;
        # create a spectrogram
        self.sp = Spectrogram(self.wf, 10, np.hanning, 5, 100)

    def test_len(self):
        self.assertEqual(len(self.wf), len(self.sp))

    def test_iter(self):
        for i in self.sp:
            self.assertEqual(len(self.sp.bins), len(i))

    def test_get(self):
        self.assertEqual(len(self.sp.bins), len(self.sp[0]))

    def test_set(self):
        self.assertFalse((self.sp[0] == np.zeros(len(self.sp.bins))).all())
        self.sp[0] = np.zeros(len(self.sp.bins))
        self.assertTrue((self.sp[0] == np.zeros(len(self.sp.bins))).all())


class TestEffects(unittest.TestCase):

    def setUp(self):
        self.f = 1000 # sample rate in Hz
        self.T = 1 / self.f # sample period
        self.t = np.arange(0,1,self.T) # time axis

        self.freq1 = 50
        self.freq2 = 100
        self.freq3 = 125
        self.freq4 = 250
        self.a = 0.5 * np.sin(2 * np.pi * self.freq1 * self.t)
        self.b = 0.5 * np.sin(2 * np.pi * self.freq2 * self.t)
        self.c = 0.5 * np.sin(2 * np.pi * self.freq3 * self.t)
        self.d = 0.5 * np.sin(2 * np.pi * self.freq4 * self.t)

        self.wf = Waveform(len(self.t), self.f)
        self.wf[:] = self.a + self.b + self.c + self.d

    def test_lowpass(self):
        lowpassed = Waveform(len(self.t), self.f)
        lowpassed[:] = self.wf._samples
        lowpassed.lowpass(110)
        bins = np.fft.fftfreq(len(self.t), self.T)
        # frequencies that should be zeroed
        mask = (np.abs(bins) < 110)
        all_fft = np.abs(self.wf.fft)
        low_fft = np.abs(lowpassed.fft)

        # tolerance threshold for verifying lowpass filter results
        tol = 2.0

        for i,v in enumerate(mask):
            if v:
                self.assertAlmostEqual(low_fft[i], all_fft[i], delta=tol)
            else:
                self.assertAlmostEqual(low_fft[i], 0.0, delta=tol)

    def test_highpass(self):

        highpassed = Waveform(len(self.t), self.f)
        highpassed[:] = self.wf._samples
        highpassed.highpass(110)
        bins = np.fft.fftfreq(len(self.t), self.T)

        # frequencies that should be zeroed
        mask = (np.abs(bins) > 110)
        all_fft = np.abs(self.wf.fft)
        high_fft = np.abs(highpassed.fft)

        # tolerance threshold for verifying highpass filter results
        tol = 2.0

        for i,v in enumerate(mask):
            if v:
                self.assertAlmostEqual(high_fft[i], all_fft[i], delta=tol)
            else:
                self.assertAlmostEqual(high_fft[i], 0.0, delta=tol)

    def test_bandpass(self):
        bandpassed = Waveform(len(self.t), self.f)
        bandpassed[:] = self.wf._samples
        bandpassed.bandpass(60, 240)
        bins = np.fft.fftfreq(len(self.t), self.T)

        # frequencies that should be zeroed
        mask = (np.abs(bins) > 90) * (np.abs(bins) < 150)
        all_fft = np.abs(self.wf.fft)
        band_fft = np.abs(bandpassed.fft)

        from matplotlib import pyplot as plt
        plt.plot(bins, all_fft)
        plt.plot(bins, band_fft)
        plt.show()

        # tolerance threshold for verifying bandpass filter results
        tol = 50

        for i,v in enumerate(mask):
            if v:
                self.assertAlmostEqual(band_fft[i], all_fft[i], delta=tol)
            else:
                self.assertAlmostEqual(band_fft[i], 0.0, delta=tol)



if __name__ == '__main__':
    unittest.main()

