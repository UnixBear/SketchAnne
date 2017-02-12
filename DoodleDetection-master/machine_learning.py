import pickle
from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import os
import numpy as np
import random

random.seed(1)

data_dir = 'lights_train'

fnames = []
for fname in os.listdir(data_dir):
    if fname.startswith('file'):
        fnames.append(fname)

random.shuffle(fnames)

raw_data = [misc.imread(os.path.join(data_dir, fname)) for fname in fnames]
img_dim = (90, 90)
data = np.zeros(((len(raw_data), img_dim[0] * img_dim[1])), dtype=np.float64)
for img_i in range(len(raw_data)):
    print('img {}'.format(img_i))
    for i in range(data.shape[1]):
        row = i / raw_data[img_i].shape[0]
        col = i % raw_data[img_i].shape[1]
        if row < raw_data[img_i].shape[0] and col < raw_data[img_i].shape[1]:
            # average rgb channels to single value
            avg = (raw_data[img_i][row][col][0] / 3.0 + raw_data[img_i][row][col][1] / 3.0 + raw_data[img_i][row][col][
                2] / 3.0)
            data[img_i][i] = avg / 255.0

faces = datasets.fetch_olivetti_faces().data

from sklearn import svm

X = data
y = [1 if fname.endswith('True.png') else 0 for fname in fnames]

# retain this fraction for validation
validation_split = 0.3
validation_index = int(X.shape[0] * (1 - validation_split))

train_X = X[:validation_index]
train_y = y[:validation_index]

test_X = X[validation_index:]
test_y = y[validation_index:]


def train():
    clf = SVC(kernel='linear')
    clf.fit(train_X, train_y)

    pickle.dump(clf, open('svc.model', 'w'))

    print ("Accuracy on training set:")
    print (clf.score(train_X, train_y))

    print ("Accuracy on validation set:")
    print (clf.score(test_X, test_y))


def predict():
    clf2 = pickle.load(open('svc.model'))
    Y_predict = clf2.predict(test_X)
    print(Y_predict)


train()

# predict()
