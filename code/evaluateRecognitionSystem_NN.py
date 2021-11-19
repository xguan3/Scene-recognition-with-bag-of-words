""" from getDictionary import get_dictionary
from getImageDistance import get_image_distance
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words

import cv2
import numpy as np
import pickle

meta = pickle.load(open('../data/traintest.pkl', 'rb'))

test_imagenames = meta['test_imagenames']
test_labels = meta['test_labels']
dictionary_list = ["Random","Harris"]
distance_list = ["euclidean", "chi2"]
imgPaths = ["../data/" + path for path in test_imagenames]

#with open('knn_harris_chi_label.pkl', 'wb') as handle:
#pickle.dump(harris_chi_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

for dict_option in dictionary_list:  
    for dist_option in distance_list:
        # need to get confusion matrix generation for all 4 combinations
        print("--------------", dict_option, "+", dist_option, "----------------")

        # define variables 
        recog_system = pickle.load (open('vision%s.pkl'%dict_option, 'rb'))
        filterBank = recog_system["filterBank"]
        train_labels = recog_system["trainLabels"]
        train_Features = recog_system["trainFeatures"]
        train_dictionary = recog_system["dictionary"]
        k = train_dictionary.shape[0]
        

        # Begin processing each of the test images
        confusion_matrix = np.zeros((8,8))

        for i, path in enumerate(imgPaths):
                print('Processing image %d/%d (%s)\r'%(i,len(imgPaths),path[8:]), end="")
                pkl_path = open('../data/%s_%s.pkl'%(path[:-4], dict_option), 'rb')
                wordMap=pickle.load(pkl_path)
                test_hist = get_image_features(wordMap, 500)

                # compare each histogram 
                nearest = 1000
                match_index = 0
                for hist_index, train_img_hist in enumerate(train_labels):
                    dist = get_image_distance(test_hist,train_img_hist,dist_option)

                    if dist < nearest:
                        nearest = dist
                        match_index = hist_index

                nearest_label = int(train_labels[match_index]) - 1 
                test_label = int(test_labels[i]) - 1
                confusion_matrix[test_label, nearest_label]+=1

        accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        print(accuracy)
        print("Resulting confusion matrix: ")
        print(confusion_matrix) 
                  """

import cv2
import pickle
import numpy as np
from getDictionary import get_dictionary
from getImageDistance import get_image_distance
from getImageFeatures import get_image_features
from getVisualWords import get_visual_words


meta = pickle.load (open('../data/traintest.pkl', 'rb'))

test_imagenames = meta['test_imagenames']


# -----fill in your implementation here --------
test_labels = meta['test_labels']
dictionary_options = ["Random","Harris"]
distance_options = ["euclidean", "chi2"]
imgPaths = ["../data/" + path for path in test_imagenames]

for dict_option in dictionary_options:  
    for dist_option in distance_options:
        # We need to run the same confusion matrix generation for all 4 combos
        print('------',dict_option,"+",dist_option,'---------')

        # The previously calculated recog system will have most of what we need
        recog_system = pickle.load (open('vision%s.pkl'%dict_option, 'rb'))
        labels = recog_system["trainLabels"]
        dictionary = recog_system["dictionary"]
        train_features = recog_system["trainFeatures"]
        k = dictionary.shape[0]

        # Begin processing each of the test images
        confusion_matrix = np.zeros((8,8))
        for i, path in enumerate(imgPaths):
                print('Processing image %d/%d (%s)\r'%(i,len(imgPaths),path[8:]), end="")
                pkl_path = open('../data/%s_%s.pkl'%(path[:-4], dict_option), 'rb')
                wordMap=pickle.load(pkl_path)
                test_hist = get_image_features(wordMap, 500)

                # compare each histogram to all of the trianing histograms
                nearest = 999
                match_index = 0
                for hist_index, train_img_hist in enumerate(train_features):
                    dist = get_image_distance(test_hist,train_img_hist,dist_option)

                    if dist < nearest:
                        nearest = dist
                        match_index = hist_index

                # Map to confusion matrix requires -1 since labels go 1-8 but python goes 0-7
                nearest_label = int(labels[match_index]) - 1 
                actual_label = int(test_labels[i]) - 1
                confusion_matrix[actual_label, nearest_label]+=1
                accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

        print(accuracy)        
        print("\nResulting confusion matrix: ")
        print(confusion_matrix)       
        print("\n") 
