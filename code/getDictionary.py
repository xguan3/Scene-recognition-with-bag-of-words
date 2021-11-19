import numpy as np
import cv2
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from createFilterBank import create_filterbank

from sklearn.cluster import KMeans

def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()
    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))
    row_counter = 0
    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv2.imread('../data/%s' % path)

        # -----fill in your implementation here --------

        filterResponses = extract_filter_responses(image, filterBank)

        if method == "Harris":
            points = [get_harris_points(filterResponses[i], alpha) for i in range(60)]
        elif method == "Random":
            points = [get_random_points(filterResponses[i], alpha) for i in range(60)]

        for j in range(alpha):
            for k in range(len(points)):
                point_x = points[k][j][0]
                point_y = points[k][j][1]
                pixelResponses[row_counter][k] = filterResponses[k][point_x, point_y]
            row_counter += 1

        # ----------------------------------------------

    dictionary = KMeans(n_clusters=K, random_state=0).fit(pixelResponses).cluster_centers_
    return dictionary
