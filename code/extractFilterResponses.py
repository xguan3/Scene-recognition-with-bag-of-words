from RGB2Lab import rgb2lab
from utils import *
import cv2 as cv
from createFilterBank import create_filterbank

def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------
    I_lab = rgb2lab(I)
    filterResponses = []
    
    for i in filterBank:
        filt_1 = imfilter(I_lab[:,:,0],i)
        filt_2 = imfilter(I_lab[:,:,1],i)
        filt_3 = imfilter(I_lab[:,:,2],i)
        filterResponses.append(filt_1)
        filterResponses.append(filt_2)
        filterResponses.append(filt_3)
    # ----------------------------------------------
    
    return filterResponses



if __name__ == "__main__":
    fb = create_filterbank()
    I = cv.imread("../data/desert/sun_adpbjcrpyetqykvt.jpg")

    responses = extract_filter_responses(I, fb)
    
    # For testing, build the sample image from each of the response indices
    for i in range(responses[1]):
        filtered_image = np.zeros((I.shape[0], I.shape[1]))
        filtered_image = filtered_image.astype(np.uint8) 
        
        for x in range(I.shape[0]):
            for y in range(I.shape[1]):
                filtered_image[x,y] = responses[x,y, i]
        
        cv.imshow(str(i),filtered_image)
        cv.waitKey(0)
        cv.destroyAllWindows()
