# -*- coding: utf-8 -*-
"""
Created on Sat May 18 11:15:01 2024

@author: afisher
"""

import sys
import numpy as np
import gc 

def return_size(df, unit='MB'):
    unit_order = {'GB':1e9, 'MB':1e6, 'B':1}
    size = round(sys.getsizeof(df)/unit_order[unit], 2)
    
    print(f'Memory usage: {size} {unit}')
    return size


def convert_types(*dfs, print_info=True, unit='MB'):
    
    print('Original:')
    for df in dfs:
        return_size(df)
        
    print('Reduced:')
    for df in dfs:
        
        for col in df:
            
            if df[col].dtype == float:
                df[col] = df[col].astype(np.float32)
                
            elif df[col].dtype == int:
                df[col] = df[col].astype(np.int32)
            
            elif df[col].dtype=='object' and df[col].nunique()<df.shape[0]*0.8:
                df[col] = df[col].astype('category')

        return_size(df, unit=unit)
    return dfs
    

def delete_from_memory(*var_names):
    gc.enable()
    
    for var_name in var_names:
        if var_name in globals():
            del globals()[var_name]
    
    gc.collect()
    
    return
    