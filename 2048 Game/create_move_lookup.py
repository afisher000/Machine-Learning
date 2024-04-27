# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:54:04 2023

@author: afisher
"""
import numpy as np
import pickle

def merge(line):
    result = [0] * len(line)
    index = 0
    for value in line:
        if value != 0:
            if result[index] == 0:
                result[index] = value
            elif result[index] == value:
                result[index] += 1 #increment by 1 on logboard
                index += 1
            else:
                index += 1
                result[index] = value
    return result


move_dict = {}
max_int = 12
for i1 in range(max_int):
    for i2 in range(max_int):
        for i3 in range(max_int):
            for i4 in range(max_int):
                line = np.array([i1, i2, i3, i4], dtype=int)
                merged = merge(line)
                
                
                move_dict[tuple(line)] = merged
                
with open('move_lookup.pkl', 'wb') as f:
    pickle.dump(move_dict, f)