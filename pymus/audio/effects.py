import numpy as np

@waveform_effect
def lowpass(wf, cuttoff_freq):
    """Use a windowed sinc function with a blackman window to apply a lowpass
    filter on a waveform object."""
    window = np.blackman(len(wf))

