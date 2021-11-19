import pickle
from getDictionary import get_dictionary


meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']

# -----fill in your implementation here --------
a = 200
k = 500
#dictionary = get_dictionary(train_imagenames, a, k, "Harris")
#pickle.dump(dictionary, open("harris_dict.pkl", "wb"))

dictionary = get_dictionary(train_imagenames, a, k, "Random")
pickle.dump(dictionary, open("random_dict.pkl", "wb"))
# ----------------------------------------------



