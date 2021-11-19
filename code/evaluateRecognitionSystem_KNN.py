from getDictionary import get_dictionary
from getImageDistance import get_image_distance
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
from sklearn.neighbors import KNeighborsClassifier
from utils import chi2dist

from scipy.spatial.distance import euclidean
 
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def chi2Dist(hist1, hist2):
    dist = chi2dist(hist1, hist2)
    return dist

def euclideanDist(hist1, hist2):
    dist = euclidean(hist1, hist2)
    return dist

meta = pickle.load(open('../data/traintest.pkl', 'rb'))

test_imagenames = meta['test_imagenames']
test_labels = meta['test_labels']

method = 'Random'
imgPaths = ["../data/" + path for path in test_imagenames]

recog_system = pickle.load(open('vision%s.pkl'%method, 'rb'))
train_features = recog_system['trainFeatures']
train_labels = recog_system['trainLabels']
dictionary = recog_system["dictionary"]
filterBank = recog_system["filterBank"]


accuracy = np.zeros(41)
best_confusion = np.zeros((8,8))
accuracy = []
confusion_list = []

for i in range(1,41):
    neighbors = KNeighborsClassifier(n_neighbors = i, metric = euclideanDist)
    neighbors.fit(train_features, train_labels)
    

    #pkl_path = open('../data/%s_%s.pkl'%(path[:-4], "Random"), 'rb')
    
    confusion_matrix = np.zeros((8,8))

    for j, path in enumerate(imgPaths):
        img = cv2.imread(path)

        pkl_path = open('../data/%s_%s.pkl'%(path[:-4], "Harris"), 'rb')
        wordMap=pickle.load(pkl_path)
        features = get_image_features(wordMap, 500)
        
        test_label = int(test_labels[j])
        classified_label = int(neighbors.predict(features.reshape(1, -1))[0])
        confusion_matrix[test_label-1, classified_label-1] = confusion_matrix[test_label-1, classified_label-1] + 1

    accuracy.append(np.trace(confusion_matrix)/np.sum(confusion_matrix)) 
    confusion_list.append(confusion_matrix)
    print(accuracy)
    print(confusion_matrix)
    print('k value: {} Accuracy: {} Method: {} + {}'.format(i, accuracy[i-1], "Harris", "euclidean"))
    print('\n')

    if i == np.argmax(test_label):
        best_confusion = confusion_matrix

best_k = np.argmax(accuracy)
best_confusion = confusion_list[best_k]
print("Best knn: {} Accuracy: {}".format(best_k,accuracy[best_k]))

plt.plot(list(range(1,41)),accuracy)
plt.show()
print("Best k number of nearest neighbors was {} with accuracy {}".format(best_k,accuracy[best_k]))
print(best_confusion)

