import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
from extractFilterResponses import extract_filter_responses
from createFilterBank import create_filterbank
from skimage.color import label2rgb

import pickle
import cv2

def get_visual_words (I, dictionary, filterBank):

    # -----fill in your implementation here --------
    filtered_I = extract_filter_responses(I, filterBank)

    H,W = I.shape[0], I.shape[1]

    wordMap = np.zeros((H,W))
    
    for col_idx in range(H):
        for row_idx in range(W):
            filter_vector = np.asarray([filtered_I[n][col_idx][row_idx] for n in range(len(filtered_I))])
            euclidean_distance = cdist(dictionary, [filter_vector], metric = 'euclidean')
            wordMap[col_idx][row_idx] = np.argmin(euclidean_distance)
    # ----------------------------------------------

    return wordMap

# test

""" if __name__ == '__main__':
    img = cv2.imread('../data/airport/sun_aesovualhburmfhn.jpg')
    dict = pickle.load(open( "dictionaryHarris.pkl", "rb" ))
    filterBank = create_filterbank()

    wordMap = get_visual_words(img, dict, filterBank)

    print(wordMap) """

""" if __name__ == '__main__':
    img = cv2.imread('../data/bedroom/sun_aaprepyzloaivblt.jpg')
    rand_dict = pickle.load(open( "dictionaryRandom.pkl", "rb" ))
    harris_dict = pickle.load(open( "dictionaryHarris.pkl", "rb" ))
    filterBank = create_filterbank()

    randomWordMap = get_visual_words(img, rand_dict, filterBank)
    harrisWordMap = get_visual_words(img, harris_dict, filterBank)

    visualize_random = label2rgb(randomWordMap, img, bg_label=0)
    visualize_harris = label2rgb(harrisWordMap, img, bg_label=0)
    #cv2.imshow(str("Random Word Map in RGB"), visualize_random)
    #cv2.imshow(str("Harris Word Mapp in RGB"), visualize_harris)

    visualize_random = cv2.cvtColor(np.float32(visualize_random), cv2.COLOR_RGB2BGR)
    visualize_harris = cv2.cvtColor(np.float32(visualize_harris), cv2.COLOR_RGB2BGR)
    cv2.imshow(str("Sample from Random dictionary in BGR"), visualize_random)
    cv2.imshow(str("Sample from Harris dictionary in BGR"), visualize_harris)

    cv2.waitKey(0)
    cv2. destroyAllWindows() """

