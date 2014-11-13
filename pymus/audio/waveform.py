import numpy as np

class Waveform:
    """Represents a single-channel waveform of a sound sample."""

    def __init__(self, length, sample_rate):
        """Construct an empty array with the given length."""
        self._samples = np.zeros(length)
        self.sample_rate = sample_rate

    def __iter__(self):
        """Allow iteration."""
        return (self[i] for i in range(len(self)))

    def __len__(self):
        """Retun the length."""
        return len(self._samples)

    def __getitem__(self, key):
        """Allows reading values using indexes."""
        try:
            return self._spawn(self._samples[key])
        except TypeError:
            # make single values iterable
            return self._spawn([self._samples[key]])

    def __setitem__(self, key, val):
        """Allows setting values using indexes capping them from -1 to 1."""
        self._samples[key] = np.clip(val, -1.0, 1.0)

    def __add__(self, num):
        """Apply gain to every sample, capping at 1.0."""
        return self._spawn(np.clip(self._samples + num, -1.0, 1.0))

    def __sub__(self, num):
        """Apply negative gain to every sample."""
        return self + -num

    def __mul__(self, val):
        """Apply a window (array input) or scale every sample (number input)"""
        return self._spawn(np.clip(self._samples * val, -1.0, 1.0))

    def __pow__(self, num):
        """Apply an exponent to every sample."""
        return self._spawn(np.clip(self._samples ** num, -1.0, 1.0))

    def _spawn(self, data):
        """Create a new waveform with the same metadata as this waveform."""
        wf = Waveform(len(data), self.sample_rate)
        wf[:] = data
        return wf

    @property
    def rms(self):
        """Calculate the root mean squared."""
        return np.sqrt(sum(self ** 2 / len(self)))

    @property
    def dBFS(self):
        """Calculate the decibels relative to full scale."""
        return 10 * np.log(self.rms)

    @property
    def fft(self):
        """Calculate the fft of this sample."""
        return np.fft.fft(self._samples)

    def resample(self, rate):
        """Resample the waveform to a new rate."""
        pass # not implemented yet

    def upsample(self, rate):
        """Upsample the waveform to a new rate."""
        if (rate < self.sample_rate):
            raise ValueError("Rate is less than the current sampling rate.")

        from fractions import Fraction

        ratio = Fraction(rate, self.sample_rate)
        new_samples = self._samples
        # insert L zeros at every M index
        for i in range(1,ratio.numerator):
            stop = len(new_samples)
            skip = i;
            indeces = np.arange(1, stop, skip)
            new_samples = np.insert(new_samples, indeces, 0)

        return new_samples
        """
        sinc = np.sinc(np.arange(-5,5))
        win_sinc = sinc * np.hanning(len(sinc))
        return np.convolve(new_samples, win_sinc)
        """

if __name__ == '__main__':
    """Testing."""
    wf = Waveform(100, 2)
    wf[0:100] = np.sin(np.linspace(0, 2 * np.pi, 100))
    print(wf.upsample(4))
