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

def check_is_major(chord, print=False):
    c = np.unique(np.sort(chord%12))
    if len(c)!=3:
        return False

    while c[0]<12:
        if np.array_equal([4,3], np.diff(c)):
            if print:
                print('is major')
            return True
        c = np.sort(c + [12, 0, 0])
    return False
    
def check_is_minor(chord, print=False):
    c = np.unique(np.sort(chord%12))
    if len(c)!=3:
        return False

    while c[0]<12:
        if np.array_equal([3,4], np.diff(c)):
            if print:
                print('is minor')
            return True
        c = np.sort(c + [12, 0, 0])
    return False
        
def check_is_diminished(chord, print=False):
    c = np.unique(np.sort(chord%12))
    if len(c)!=3:
        return False

    while c[0]<12:
        if np.array_equal([3,3], np.diff(c)):
            if print:
                print('is diminished')
            return True
        c = np.sort(c + [12, 0, 0])
    return False
        


for j in range(33):
    c = cs[j]
    if not check_is_major(c):
        if not check_is_minor(c):
            if not check_is_diminished(c):
                print(cs[j])








# %%
