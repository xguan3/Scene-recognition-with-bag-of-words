import numpy as np


def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------

    histogram = np.zeros (dictionarySize)

    for x in range(wordMap.shape[0]):
        for y in range(wordMap.shape[1]):
            idx = int(wordMap[x,y])
            histogram[idx]+=1
    
    histogram = histogram/np.sum(histogram)
    # ----------------------------------------------
    
    return histogram
