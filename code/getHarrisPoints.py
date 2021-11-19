import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter


def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------

    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = imfilter(I, kernel_x)
    Iy = imfilter(I, kernel_y)
    Ixx = Ix * Ix
    Ixy = Iy * Ix
    Iyy = Iy * Iy

    height = I.shape[0]
    width = I.shape[1]
    offset = (np.floor(kernel_x.shape[0]/2)).astype(int)

    harris_response = []

    # Harris response calculation
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sxx = np.sum(Ixx[y-offset:y+1+offset, x-offset:x+1+offset])
            Syy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
            Sxy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])

            # Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy * Sxy)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            harris_response.append([y, x, r])

    harris_response_sorted = sorted(harris_response, key = lambda x:x[2], reverse = True)

    # Find edges and corners using R
    points = []
    for response in harris_response_sorted[0:alpha]:
        y, x, r = response
        if r > 0:
            points.append([y, x])
    # ----------------------------------------------
    
    return points

