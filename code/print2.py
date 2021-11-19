from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from createFilterBank import create_filterbank
from getVisualWords import get_visual_words
from RGB2Lab import rgb2lab
from skimage.color import label2rgb

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pickle

import os.path

# -------------- 1.2 ------------------
my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../data/auditorium/sun_abfkbtrejmpugmic.jpg")

imagecv = cv.imread(path)
imagecv = cv.cvtColor(imagecv, cv.COLOR_BGR2RGB)
image = plt.imread(path)
implot = plt.imshow(image)

filterBank = create_filterbank()
filter_response = extract_filter_responses(imagecv, filterBank)
count = 1
for flt in filter_response:
    count = count + 5
    if count > 23 and count < 30:
        cv.imshow('image', flt)
        k = cv.waitKey(50000)
    print(count)

#-------------- 2.1 -----------------------
imgagecv2 = cv.imread('../data/bedroom/sun_aaprepyzloaivblt.jpg')
rand_dict = pickle.load(open( "dictionaryRandom.pkl", "rb" ))
harris_dict = pickle.load(open( "dictionaryHarris.pkl", "rb" ))
filterBank = create_filterbank()

randomWordMap = get_visual_words(imgagecv2, rand_dict, filterBank)
harrisWordMap = get_visual_words(imgagecv2, harris_dict, filterBank)

visualize_random = label2rgb(randomWordMap, imgagecv2, bg_label=0)
visualize_harris = label2rgb(harrisWordMap, imgagecv2, bg_label=0)

cv.imshow(str("Random Word Map in RGB"), visualize_random)
cv.imshow(str("Harris Word Mapp in RGB"), visualize_harris)

visualize_random = cv.cvtColor(np.float32(visualize_random), cv.COLOR_RGB2BGR)
visualize_harris = cv.cvtColor(np.float32(visualize_harris), cv.COLOR_RGB2BGR)
cv.imshow(str("Sample from Random dictionary in BGR"), visualize_random)
cv.imshow(str("Sample from Harris dictionary in BGR"), visualize_harris)

cv.waitKey(1000)
#cv. destroyAllWindows()
