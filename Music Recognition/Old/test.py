# %%
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import utils_model as um
import utils_io as uio
import utils_music_munging as umm
import utils_image_processing as uip
import utils_music_theory as umt
import cv2 as cv
from scipy import ndimage
import os
import pickle
all_cols = [
    'state', 'area','width','height','aspectratio','extent','solidity','angle'
    ]
data_cols = [
    'area','width','height','aspectratio','extent','solidity','angle'
    ]
# Load file
df = pd.read_csv('training_data_filling.csv')
df = df[df.area<2]

color_map = {'\r':'red','1':'green'}
df.plot.scatter('area','extent', color=df.state.map(color_map))


def normalize(series):
    return (series-series.mean())/series.std()

df[data_cols] = df[data_cols].apply(normalize)
df.agg(func=[np.mean, np.std])

# Pick first "true"
df = df.reset_index(drop=False)
idx = np.where(df.state=='1')[0][0]
alldists = df[data_cols] - df.loc[idx, data_cols] 

dists = (alldists**2).sum(axis=1).sort_values()
results = pd.Series(
    data = df.state[dists.index].values,
    index = dists.values
)
# results['state'] = df.state[results.index]




# %%
