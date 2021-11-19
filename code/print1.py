import cv2 as cv
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
import matplotlib.pyplot as plt
from getHarrisPoints import get_harris_points
from getRandomPoints import get_random_points

import os.path

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, "../data/campus/sun_abslhphpiejdjmpz.jpg")

#path = r"\Users\guanxinran\Desktop\Multimedia\MM811\Assignment 1\assignment_mm811\data\airport"
imagecv = cv.imread(path)
imagecv = cv.cvtColor(imagecv, cv.COLOR_BGR2RGB)
image = plt.imread(path)
implot = plt.imshow(image)

# Plot Random/harris Points
points = get_harris_points(imagecv, 500, 0.05)
count = 0
for p in points:
    plt.scatter(p[count+1],p[count], c='r', s=2)
plt.show()
k = cv.waitKey(2000)

