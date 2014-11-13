import numpy as np
from pymus.audio.waveform import Waveform

class Spectrogram:
    """A time-domain 3d representation of sound with frequency on the y-axis
    and amplitude on the z axis."""

    def __init__(self, wf, win_len, win_func, step, samp_rate):
        self.source = wf
        self.window = win_func(win_len)
        self.step = step
        self.sample_rate = samp_rate
        self.bins = np.fft.fftfreq(len(self.window), 1.0/self.sample_rate)
        self._stft = self._build_stft()

    def _build_stft(self):
        """Creates a short-time fourier transform of the waveform."""
        mask = range(len(self.window), len(self.source), self.step)
        stft = np.zeros((len(self.source), len(self.bins)))
        for i,n in enumerate(mask):
            segment = self.source[n - len(self.window):n]
            windowed = segment * self.window
            stft[i,:] = np.abs(windowed.fft)

        return stft

    def __len__(self):
        """Return the length of this spectrogram."""
        return len(self._stft)

    def __iter__(self):
        """Allow iteration."""
        return (self[i] for i in range(len(self)))

    def __getitem__(self, key):
        """Allows reading values using indexes."""
        return self._stft[key]

    def __setitem__(self, key, value):
        """
        Allows setting values using indeces.
        Note that len(value) must equal len(bins)
        """
        self._stft[key] = value

    def to_waveform(self):
        """Reconstruct a waveform from the STFT."""
        wf = Waveform()
        mask = range(len(self.window), len(self.source), self.step)
        for i,n in enumerate(mask):
            fft = np.fft.ifft(self.spec[i])
            start = self.step * i
            stop = self.step * i + len(self.window)
            wf[start:stop] += fft

        return wf
