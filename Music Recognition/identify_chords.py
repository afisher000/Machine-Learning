# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read files
notes = pd.read_csv('notes.csv')


cs = []
notes = notes.set_index(['line', 'x'])
for index in notes.index.unique():
    pitches = notes.loc[index, 'pitch'].values
    if len(pitches)==4:
        cs.append(pitches)

# %%
        
def identify_chord(chord, disp=True):
    chord_types = {
        'major':[4,3], 
        'minor':[3,4],
        'diminished':[3,3],
        'augmented':[4,4]
    }

    pitch_to_string = { 
        0:'C',
        1:'C#/Db',
        2:'D',
        3:'D#/Eb',
        4:'E',
        5:'F',
        6:'F#/Gb',
        7:'G',
        8:'G#/Ab',
        9:'A',
        10:'A#/Bb',
        11:'B'
    }

    # Ensure chord is np.array and sorted
    chord = np.sort(np.array(chord))

    # Mod to octave
    c = np.unique(np.sort(chord%12))
    if len(c)!=3:
        if disp:
            print('Not 3 notes in chord')
        return False

    # Check inversions against 4 chord_types
    while c[0]<12:
        for chord_type, pattern in chord_types.items():
            if np.array_equal(pattern, np.diff(c)):
                if disp:
                    chord_key = c[0]
                    root = chord[0]
                    chord_string = pitch_to_string[chord_key] + chord_type
                    if chord_key != root%12:
                        chord_string += '/'+pitch_to_string[root%12]
                    print(chord_string)
                return
        # Invert
        c = np.sort(c+[12,0,0])
    
    # If not identified
    print(chord)
    return

for j in range(len(cs)):
    c = cs[j]
    identify_chord(c)









# %%
