import os
import numpy as np
import pandas as pd

from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

input_data_path = 'data'

with open(os.path.join(input_data_path, "train_neg.txt")) as f:
    text_lines_neg = f.read().splitlines()
train_neg = np.array(text_lines_neg)

with open(os.path.join(input_data_path, "train_pos.txt")) as f:
    text_lines_pos = f.read().splitlines()
train_pos = np.array(text_lines_pos)

with open(os.path.join(input_data_path, "test_data.txt")) as f:
    text_lines_test = f.read().splitlines()
X_test = np.array(list(map(lambda str: str.split(",")[1],text_lines_test)))

X = np.concatenate((train_neg, train_pos))
y = np.concatenate((np.zeros(train_neg.shape[0]), np.ones(train_pos.shape[0])))
df = pd.DataFrame(X, columns=["text"])
df['label'] = y

train_df = df.sample(frac=0.9, random_state=0)
dev_df = df.drop(train_df.index)
test_df = pd.DataFrame(X_test, columns=["text"])
test_df['label'] = None

flair_data_path = 'data/transformed_data'

train_df.to_csv(os.path.join(flair_data_path, 'train.csv'), index=False)
dev_df.to_csv(os.path.join(flair_data_path, 'dev.csv'), index=False)
#test_df.to_csv(os.path.join(flair_data_path, 'test.csv'), index=False)


# column format indicating which columns hold the text and label(s)
column_name_map = {0: "text", 1: "label",}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus: Corpus = CSVClassificationCorpus(flair_data_path,
                                         column_name_map,
                                         skip_header=True,
                                         delimiter=',',
)
print(corpus)