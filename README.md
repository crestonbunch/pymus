# pymus

Pymus is a python audio library currently under development meant to perform all audio related tasks.

## Quickstart

Apply a lowpass filter:

```
import pymus.file.wav
import pymus.audio.effects

# open a two-channel .wav file
a,b = pymus.file.wav.load(‘tests/data/test1.wav’)

# cutoff frequencies above 500 Hz
a.lowpass(500)
b.lowpass(500)

# save as a new .wav file
pymus.file.wav.save(‘out.wav’, a, b)

```
