import unittest
import numpy as np
from pymus.audio.waveform import Waveform
from pymus.audio.spectrogram import Spectrogram

class TestWaveform(unittest.TestCase):

    def setUp(self):
        # create 500 samples of a sine wave at 1 Hz
        self.samples = np.sin(np.linspace(0, 5 * 2 * np.pi, 500))
        # 100 samples / sec
        self.wf = Waveform(len(self.samples), 100) 
        self.wf[:] = self.samples;

    def test_init(self):
        self.assertEqual(500, len(self.wf))

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

    def test_pow(self):
        powed = self.wf ** 2
        self.assertEqual(powed._samples[100], self.wf._samples[100] ** 2)
        self.assertTrue(
            (powed._samples == self.wf._samples ** 2).any()
        )
        self.assertTrue((powed._samples <= 1.0).all())
        self.assertTrue((powed._samples >= -1.0).all())

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

if __name__ == '__main__':
    unittest.main()

