import wave
import struct
import numpy as np
from pymus.audio.waveform import Waveform

def load(path):
    """Load a wave file from the given path."""

    with wave.open(path, 'rb') as f:

        def _build_wf(samples):
            """Converts samples into a waveform."""
            # normalize all samples
            max_amp = 2**(8 * f.getsampwidth() - 1) - 1
            wf = Waveform(len(samples), f.getframerate())
            wf[:] = samples / max_amp
            return wf

        # buffer all frames
        buff = f.readframes(f.getnframes())
        # put buffer into a numpy array (wav format uses little endian)
        frames = np.frombuffer(buff, dtype='<i2')
        # separate channels
        channels = frames.reshape(-1, f.getnchannels()).transpose()
        # pack channels into separate waveforms
        out = [_build_wf(c) for c in channels]
        return out

def save(path, channel_a, channel_b = None):
    """Saves up to 2 channels as a wav file."""

    out = wave.open(path, 'w')
    out.setparams((1,2,channel_a.sample_rate,0,'NONE','not compressed'))

    samples = channel_a._samples;

    if channel_b is not None:
        if len(channel_a) != len(channel_b):
            raise ValueError('Channels must be the same length.')
        if channel_a.sample_rate != channel_b.sample_rate:
            raise ValueError('Channels must have the same sample rate.')

        # pack both channels together
        samples = np.empty((len(channel_a) + len(channel_b)))
        samples[0::2] = channel_a._samples
        samples[1::2] = channel_b._samples

        out.setnchannels(2)

    max_amp = 2**(8 * 2 - 1) - 1
    # prepare samples for wav format
    samples = np.round(samples * max_amp).astype('int16')
    # process one second at a time
    skip = channel_a.sample_rate
    # process each chunk and append to file output
    for i in range(0, len(samples), skip):
        chunk = samples[i:i+skip]
        fmt = 'h' * len(chunk)
        b = struct.pack(fmt, *chunk)
        out.writeframes(b)

    out.close()

