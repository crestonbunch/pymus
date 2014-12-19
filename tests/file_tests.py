import unittest
import os
import numpy as np
import pymus.file.wav
from pymus.audio.waveform import Waveform
from tempfile import NamedTemporaryFile

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

class TestWaveFile(unittest.TestCase):

    def test_load(self):
        wf = pymus.file.wav.load(os.path.join(DATA_DIR, 'test1.wav'))
        self.assertEqual(320303, len(wf[0]))
        self.assertEqual(320303, len(wf[1]))
        self.assertEqual(2, len(wf))
        self.assertEqual(32000, wf[0].sample_rate)
        self.assertEqual(32000, wf[1].sample_rate)

    def test_save(self):
        a = Waveform(10000,1000)
        b = Waveform(10000,1000)
        a[:] = np.random.rand(10000)
        b[:] = np.random.rand(10000)

        with NamedTemporaryFile('w+b', suffix='.wav') as f:
            pymus.file.wav.save(f.name,a,b)

if __name__ == '__main__':
    unittest.main()

