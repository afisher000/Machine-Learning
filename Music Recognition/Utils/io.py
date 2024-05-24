import fitz
import numpy as np
import cv2 as cv
from scipy.io.wavfile import read, write
from scipy.interpolate import interp1d
import os
import pandas as pd

song_parameter_file = 'song_params.csv'

def convert_to_jpg(file):
    # If already jpg, return
    if file.endswith('.jpg'):
        return file

    # If pdf, convert to jpg
    if file.endswith('.pdf'):
        dest = file[:-4]+'.jpg'
        pdf2jpg(file, dest=dest)
        return dest


def pdf2jpg(file, dest=None):
    doc = fitz.open(file)
    page = doc[0]
    pix = page.get_pixmap(dpi=600)
    if dest is None:
        pix.save(file[:-4]+'.jpg')
    else:
        pix.save(dest)
    return

def import_song(song_file, threshold=200):
    if not os.path.exists(song_file):
        raise ValueError('Song file not found.')
        
    if song_file.endswith('pdf'):
        pdf2jpg(song_file)
        song_file = song_file[:-4] + '.jpg'

    # Read as grayscale
    grayscale_img = cv.imread(song_file, cv.IMREAD_GRAYSCALE)
    
    # Threshold to black and white
    max_val, min_val = 255, 0
    _, binary_img = cv.threshold(grayscale_img, threshold, max_val, min_val)
    
    # Save song parameters
    save_song_params(binary_img)
    return binary_img

def save_song_params(img):
    # Get data from image
    is_line = img.sum(axis=1)/img.shape[1]<127
    line_starts = np.logical_and(~is_line[:-1], is_line[1:]).nonzero()[0]+1
    line_ends = np.logical_and(is_line[:-1], ~is_line[1:]).nonzero()[0]+1
    line_sep = int((line_starts[4]-line_starts[0])//4)
    n_lines = int(len(line_starts)//5)
    line_thickness = (line_ends-line_starts).max()

    # Save to file
    params = pd.Series()
    params['line_sep'] = line_sep
    params['n_lines'] = n_lines
    params['line_thickness'] = line_thickness
    params['margin'] = 5
    params['line_height'] = (4+2*params.margin)*params.line_sep
    params.to_csv(song_parameter_file)
    return params

def get_song_params(names):
    params = pd.read_csv(song_parameter_file, index_col=0).squeeze()
    if isinstance(names, list):
        if len(names)==1:
            return params[names].values[0]
        else:
            return params[names].values
    else:
        return params[names]
        

def show_image(img, reduce=0):
    temp = img.copy()
    for j in range(abs(reduce)):
        temp = cv.pyrDown(temp) if reduce>0 else cv.pyrUp(temp)
    cv.imshow('test', temp)
    key = cv.waitKey(0)
    cv.destroyAllWindows()

    # Check for zooming
    if key==61: #plus sign to zoom in
        show_image(img, reduce=reduce-1)
    elif key==45: #minus sign to zoom out
        show_image(img, reduce=reduce+1)
    return

def write_to_WAV(notes):
    # Read files
    sample_rate, guitar_note = read('guitarnote_A#.wav')
    bps = 2
    # Compute time and sample_rate for notes
    notes['sample_rate'] = notes.pitch.apply(lambda x: sample_rate*2**((x+2)/12)).astype(int)
    
    
    # Define audio array
    audio_length = (notes.beat.max() + 4) / bps
    n_samples = int(audio_length * sample_rate)
    audio = np.zeros(n_samples)
    time = np.linspace(0, audio_length, n_samples)
    
    # Add 1 second clips for each note
    for index, row in notes.iterrows():
        f = interp1d(row.beat/bps + np.arange(row.sample_rate)/row.sample_rate, guitar_note[:row.sample_rate], 
        'linear', bounds_error = False, fill_value=0)
        audio += f(time)
        
    # Scale audio to 32000 for int16
    audio *= 32000/np.abs(audio).max()
    audio = audio.astype(np.int16)
    
    # Write to file
    write('test.wav',sample_rate, audio)
    
