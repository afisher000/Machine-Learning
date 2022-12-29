# %%
import pandas as pd
import numpy as np
from scipy.io.wavfile import read, write
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Read files
notes = pd.read_csv('notes.csv')
sample_rate, guitar_note = read('guitarnote_A#.wav')

# Compute time and sample_rate for notes
sec_per_line = 15
samples_per_line = sec_per_line*sample_rate
line_length = notes.x.max()+100
notes['time'] = (notes.line + notes.x/line_length)*sec_per_line
notes['sample_rate'] = notes.pitch.apply(lambda x: sample_rate*2**((x+2)/12)).astype(int)

# Scale amplitude to bring out base notes
octave_amps = [2, 1.5, 1, 0.5]
notes['amplitude'] = notes.pitch.apply(lambda x: 1 - 0.7*(x+2)/12)

# Define audio array
audio_length_sec = sec_per_line * (notes.line.max()+1)
audio_length_samples = int(audio_length_sec * sample_rate)
audio = np.zeros(audio_length_samples)
time = np.linspace(0, audio_length_sec, audio_length_samples)

# Add 1 second clips for each note
for index, row in notes.iterrows():
    f = interp1d(row.time + np.arange(row.sample_rate)/row.sample_rate, guitar_note[:row.sample_rate], 
    'linear', bounds_error = False, fill_value=0)
    audio += row.amplitude*f(time)
    
# Scale audio to 32000 for int16
audio *= 32000/np.abs(audio).max()
audio = audio.astype(np.int16)

# Write to file
write('test.wav',sample_rate, audio)



# %%
