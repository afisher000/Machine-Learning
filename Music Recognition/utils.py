import fitz
import numpy as np

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

def get_ellipse_goodness(c, ellipse):
    (x0,y0), (a,b), theta = ellipse
    c = c.squeeze(1)
    r = c-[x0,y0]

    angle = theta*np.pi/180 - np.arctan2(r[:,1],r[:,0])
    c_dist = np.sqrt(np.power(r,2).sum(axis=1))
    e_dist = np.sqrt((a/2*np.cos(angle))**2 + (b/2*np.sin(angle))**2)
    return np.max(np.abs(c_dist-e_dist))