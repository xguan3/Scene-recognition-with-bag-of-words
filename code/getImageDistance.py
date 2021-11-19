from utils import chi2dist
from scipy.spatial.distance import euclidean
import numpy as np

def get_image_distance(hist1, hist2, method):
    if method == "euclidean":
        dist = euclidean(hist1, hist2)
    elif method == "chi2":
        dist = chi2dist(hist1, hist2)
    return dist
