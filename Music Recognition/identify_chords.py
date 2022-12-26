# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read files
def identify_chords(notes, key_type):

    cs = []
    notes = notes.set_index(['line', 'x'])
    for index in notes.index.unique():
        pitches = notes.loc[index, 'pitch'].values
        cs.append(pitches)

    
    def identify_chord(chord, disp=True):
        chord_types = { 
            '':{0, 4, 7},
            'm':{0, 3, 7},
            'dim':{0, 3, 6},
            'aug':{0, 4, 8},
            '7':{0, 4, 7, 10},
            'maj7':{0, 4, 7, 11},
            'm7':{0, 3, 7, 10},
            'sus2':{0, 2, 4, 7},
            'sus4':{0, 4, 5, 7}
        }
    
        pitch_to_string = { 
            'flat':{
                0:'C', 1:'Db', 2:'D', 3:'Eb', 4:'E',  5:'F', 
                6:'Gb', 7:'G', 8:'Ab', 9:'A', 10:'Bb', 11:'B'
                },
            'sharp':{        
                0:'C', 1:'C#', 2:'D', 3:'D#', 4:'E', 5:'F',
                6:'F#', 7:'G', 8:'G#', 9:'A', 10:'A#', 11:'B'
                 }
        }
        
        # Ensure chord is np.array and sorted
        chord = np.sort(np.array(chord))
        
        # Mod to octave
        c = np.unique(np.sort(chord%12))
        
        if len(c)<3:
            # print('Not 3 unique notes')
            return
        
        while c[0]<12:
            # Check against chord patterns 
            for chord_type, pattern in chord_types.items():
                if set(c-c[0]).issubset(pattern):
                    chord_key = pitch_to_string[key_type][c[0]]
                    root = pitch_to_string[key_type][chord[0]%12]
                    
                    # Build chord_string
                    chord_string = chord_key + chord_type
                    if root != chord_key:
                        chord_string += '/' + root
                    print(chord_string)
                    return
            
            # Invert and check again
            c[0] += 12
            c = np.sort(c)
            
        print(f'No match for {chord}')
        return
                    
    
    
    for j in range(len(cs)):
        c = cs[j]
        identify_chord(c)







# %%
