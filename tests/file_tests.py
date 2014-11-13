import unittest
import os
from pymus.file.sound import WaveFile

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

class TestWaveFile(unittest.TestCase):

    def test_load(self):
        test = WaveFile(os.path.join(DATA_DIR, 'test1.wav'))
        self.assertEqual(320303, test.num_frames)
        self.assertEqual(2, test.num_channels)
        self.assertEqual(32000, test.frame_rate)
        self.assertEqual(2, test.sample_width)
        
if __name__ == '__main__':
    unittest.main()

