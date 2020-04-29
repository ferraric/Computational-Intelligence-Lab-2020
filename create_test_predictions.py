import os
import numpy as np
import pandas as pd

from flair.data import Sentence
from flair.models import TextClassifier

input_data_path = 'data'
model_path = 'models/flair_tutorial/best-model.pt'

with open(os.path.join(input_data_path, "test_data.txt")) as f:
    text_lines_test = f.read().splitlines()

X_test = []
for line in text_lines_test:
    cleaned_line = "".join(line.split(",")[1:])
    sentence = Sentence(cleaned_line)
    X_test.append(sentence)


classifier = TextClassifier.load(model_path)

result = list(map(lambda s: classifier.predict(s), X_test))

predicted_labels = []
for i, sentence in enumerate(X_test):
    if i % 1000 == 0:
        print("prediction", i, "out of", len(X_test))
    label = int(float(sentence.labels[0].value))
    predicted_labels.append(label)

predicted_labels = np.array(predicted_labels)
predicted_labels[predicted_labels == 0] = -1
submission = pd.DataFrame(predicted_labels, columns=['Prediction'])
submission.insert(0, 'Id', np.arange(1, submission.shape[0]+1))
submission.to_csv(os.path.join(input_data_path, 'submission.csv'), index=False, header=True)