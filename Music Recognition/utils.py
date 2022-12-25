import fitz
import numpy as np
import cv2 as cv

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

def show_image(img, reduce=0):
    temp = img.copy()
    for j in range(reduce):
        temp = cv.pyrDown(temp)
    cv.imshow('test', temp)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return

def get_staffline_separation(img, print_result=True):
    is_line = (img.sum(axis=1)/img.shape[1]) < 127
    is_line_start = np.logical_and(~is_line[:-1], is_line[1:])
    line_starts = np.where(is_line_start)[0]
    line_sep = np.diff(line_starts[:5]).mean()
    if print_result:
        print(f'Line Separation: {line_sep:.1f}')
    return line_sep