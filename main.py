import os
import numpy as np
import time

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score
from flair.models import TextClassifier
from flair.data import Sentence

data_path = 'data'

with open(os.path.join(data_path, "train_neg.txt")) as f:
    text_lines_neg = f.read().splitlines()
train_neg = np.array(text_lines_neg)

with open(os.path.join(data_path, "train_pos.txt")) as f:
    text_lines_pos = f.read().splitlines()
train_pos = np.array(text_lines_pos)

classifier = TextClassifier.load('en-sentiment')

def predict(line):
    sentence = Sentence(line)
    classifier.predict(sentence)
    return sentence.labels[0].value == 'POSITIVE'

X = np.concatenate((train_neg, train_pos))
y = np.concatenate((np.zeros(train_neg.shape[0]), np.ones(train_pos.shape[0])))

subsample_factor = 0.001
subset_indices = np.random.choice(range(X.shape[0]), size=int(X.shape[0]*subsample_factor), replace=False)
assert X.shape[0] == y.shape[0]
X_subset = X[subset_indices]
y_subset = y[subset_indices]

folds = 5
kf = ShuffleSplit(n_splits=folds, random_state=0)
scores = np.zeros(folds)
for i, (train, test) in enumerate(kf.split(X_subset)):
    start_time = time.time()
    print("fold ", i+1, "/", folds)
    # no training required
    X_test = X_subset[test]
    y_test = y_subset[test]

    y_pred = np.array(list(map(predict, X_test)))
    scores[i] = accuracy_score(y_test, y_pred)
    print("iteration took:", time.time() - start_time)


print("scores: ", scores)