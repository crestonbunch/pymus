import wave
import numpy as np
from pymus.audio.waveform import Waveform

class WaveFile:
    """Open and read wave file formats."""

    def __init__(self, filename):
        with wave.open(filename, 'rb') as f:
            self.num_channels = f.getnchannels()
            self.num_frames = f.getnframes()
            self.sample_width = f.getsampwidth()
            self.frame_rate = f.getframerate()
            buff = f.readframes(self.num_frames)
            self.frames = np.frombuffer(buff, dtype='<i2')

    def to_waveform(self):
        """Return a waveform representation of this file. Returns a
        multi-dimensional array, each channel is its own row."""
        channels = self.frames.reshape(-1, self.num_channels).transpose()
        out = []
        for i in channels:
            wf = Waveform(self.num_frames, self.frame_rate)
            wf[0:len(i)] = i / self._max_amp() # normalize samples from 0.0-1.0
            out.append(wf)

        return out

    def _max_amp(self, signed=True):
        """Return the maximum amplitude based on the sample width."""
        if signed:
           return 2**(8 * self.sample_width - 1) - 1
        else:
            return 2**(8 * self.sample_width) - 1
