import cv2
import pickle
import numpy as np
from createFilterBank import create_filterbank
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words

def build_system (inputDict, outputDict):
    filterbank = create_filterbank()
    inputdict = pickle.load(open(inputDict + ".pkl", "rb" ))
    
    meta = pickle.load (open('../data/traintest.pkl', 'rb'))
    train_imagenames = meta['train_imagenames']

    len_dict = len(inputdict) 
    len_image = len(train_imagenames)
    
    train_labels = meta['train_labels']
    dictionary_list = ["Random","Harris"]

    # get histogram 
    train_Features = np.ndarray((len_image,len_dict))
    imgPaths = ["../data/" + path for path in train_imagenames]
    for i, path in enumerate(imgPaths):    
        print('Processing image %d/%d\r'%(i,len_image), end="")
        #img = cv2.imread(path)
        pkl_path = open('../data/%s_%s.pkl'%(path[:-4], "Harris"), 'rb')
        wordMap=pickle.load(pkl_path)
        test_hist = get_image_features(wordMap, 500)
        #wordMap = get_visual_words(img, inputdict, filterbank)
        train_Features[i] = test_hist

    print("progress is done")

    #write a pickle file with below variables
    vision = { "dictionary": inputdict, "filterBank": filterbank, "trainFeatures" : train_Features,"trainLabels" : train_labels}
    pickle.dump(vision, open(outputDict + ".pkl", 'wb'))

    return vision

if __name__ == '__main__':
    #print("------------ Building recog system for Random dictionary ---------------")
    #build_system("dictionaryRandom", "visionRandom")

    print("------------ Building recog system for Harris dictionary ------------ ")
    build_system("dictionaryHarris", "visionHarris")