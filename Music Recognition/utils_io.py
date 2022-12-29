import fitz
import numpy as np
import cv2 as cv
from scipy.io.wavfile import read, write
from scipy.interpolate import interp1d

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

def import_song(song_file):
    if song_file.endswith('pdf'):
        pdf2jpg(song_file)
        song_file = song_file[:-4] + '.jpg'

    # Read as grayscale
    grayscale_img = cv.imread(song_file, cv.IMREAD_GRAYSCALE)
    
    # Threshold to black and white
    max_val, min_val, threshold = 255, 0, 127
    _, binary_img = cv.threshold(grayscale_img, threshold, max_val, min_val)
    return binary_img


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
    
    # Compute time and sample_rate for notes
    sec_per_line = 15
    samples_per_line = sec_per_line*sample_rate
    line_length = notes.x.max()+100
    notes['time'] = (notes.line + notes.x/line_length)*sec_per_line
    notes['sample_rate'] = notes.pitch.apply(lambda x: sample_rate*2**((x+2)/12)).astype(int)
    
    
    # Define audio array
    audio_length_sec = sec_per_line * (notes.line.max()+1)
    audio_length_samples = int(audio_length_sec * sample_rate)
    audio = np.zeros(audio_length_samples)
    time = np.linspace(0, audio_length_sec, audio_length_samples)
    
    # Add 1 second clips for each note
    for index, row in notes.iterrows():
        f = interp1d(row.time + np.arange(row.sample_rate)/row.sample_rate, guitar_note[:row.sample_rate], 
        'linear', bounds_error = False, fill_value=0)
        audio += f(time)
        
    # Scale audio to 32000 for int16
    audio *= 32000/np.abs(audio).max()
    audio = audio.astype(np.int16)
    
    # Write to file
    write('test.wav',sample_rate, audio)
    
