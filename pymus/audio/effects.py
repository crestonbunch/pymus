import numpy as np
from pymus.audio.waveform import waveform_effect

@waveform_effect
def lowpass(wf, cutoff_freq, transition=0.01):
    """Applies a lowpass filter to a waveform.

    Uses a blackman-windowed sinc filter.

    Parameters:
        cutoff_freq: The maximum frequency to let through.
        transition: The transition band width as a fraction of the sample rate.
    """
    win_len = 4 / transition + 1
    # frequency must be represented as a ratio of the sample rate
    freq_ratio = 2 * cutoff_freq / wf.sample_rate

    # use a blackman window on the sinc function
    window = np.blackman(win_len)
    # the filter kernel
    kernel = np.sinc(freq_ratio * np.arange(-win_len//2,win_len//2)) * window
    # normalize the kernel
    kernel /= sum(kernel)

    new_samples = np.convolve(wf._samples, kernel, 'same')
    wf[:] = new_samples

@waveform_effect
def highpass(wf, cutoff_freq, transition=0.01):
    """Applies a highpass filter to a waveform."""

    # window length must be odd
    win_len = 4 / transition + 1
    # frequency must be represented as a ratio of the sample rate
    freq_ratio = 2 * cutoff_freq / wf.sample_rate

    # use a blackman window on the sinc function
    window = np.blackman(win_len)
    # the filter kernel
    kernel = np.sinc(freq_ratio * np.arange(-win_len//2,win_len//2)) * window
    # normalize the kernel
    kernel /= sum(kernel)

    # construct a high pass kernel from a lowpass kernel
    kernel = -kernel # reverse the sign of every point
    kernel[len(kernel)//2+1] += 1 # add one to the center point

    new_samples = np.convolve(wf._samples, kernel, 'same')
    wf[:] = new_samples

@waveform_effect
def bandpass(wf, low_cutoff, high_cutoff, low_trans=0.01, high_trans=0.01):
    """Apply a band pass by using a lowpass and highpass filter."""
    wf.lowpass(high_cutoff, low_trans)
    wf.highpass(low_cutoff, low_trans)
