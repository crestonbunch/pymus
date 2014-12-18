import numpy as np
from fractions import Fraction

def waveform_effect(fn, name=None):
    """Decorator for adding effects to Waveform objects."""
    if name is None:
        name = fn.__name__

    setattr(Waveform, name, fn)
    return fn

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

    def __str__(self):
        """Present this waveform in a human-readable string."""
        return str(self._samples)

    def _spawn(self, data):
        """Create a new waveform with the same metadata as this waveform."""
        wf = Waveform(len(data), self.sample_rate)
        wf[:] = data
        return wf

    @property
    def mean(self):
        """Calculate the statistical mean of all amplitudes."""
        return np.mean(self._samples)

    @property
    def std(self):
        """Calculate the standard deviation from the mean for all amplitudes."""
        return np.std(self._samples)

    @property
    def rms(self):
        """Calculate the root mean squared."""
        return np.sqrt(np.sum([x**2 for x in self._samples]) / len(self))

    @property
    def fft(self):
        """Calculate the fft of this sample."""
        return np.fft.fft(self._samples)

    def resample(self, rate):
        """Resample the waveform to a new rate.

        Uses the Wittaker-Shannon interpolation formula.
        TODO: this is really slow, surely there's a better way
        """

        ratio = Fraction(rate, self.sample_rate)
        # sampling period
        T = 1 / self.sample_rate
        # length of the time axis
        l = self.sample_rate / len(self)
        new_samples = []
        for t in np.linspace(0,round(l),rate):
            s = [x * np.sinc((t - n*T) / T) for n,x in enumerate(self._samples)]
            x = sum(s)
            new_samples.append(x)

        # put the samples in a new waveform
        out = self._spawn(new_samples)
        out.sample_rate = rate
        return out
