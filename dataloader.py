import tensorflow as tf
import os
import re
import pandas as pd


class DataLoader:
    DATA_COLUMN = 'sentence'
    LABEL_COLUMN = 'polarity'

    def __init__(self, path):
        self.path = os.path.join('./datasets/', path)

    def load_twitter_dataset(self, full=True):
        data = {}
        data[DataLoader.DATA_COLUMN] = []
        data[DataLoader.LABEL_COLUMN] = []
        if full:
            with tf.io.gfile.GFile(os.path.join(self.path, 'train_neg_full.txt'), 'r') as f:
                for line in f:
                    data[DataLoader.DATA_COLUMN].append(line)
                    data[DataLoader.LABEL_COLUMN].append(0)
            with tf.io.gfile.GFile(os.path.join(self.path, 'train_pos_full.txt'), 'r') as f:
                for line in f:
                    data[DataLoader.DATA_COLUMN].append(line)
                    data[DataLoader.LABEL_COLUMN].append(1)
        else:
            with tf.io.gfile.GFile(os.path.join(self.path, 'train_neg.txt'), 'r') as f:
                for line in f:
                    data[DataLoader.DATA_COLUMN].append(line)
                    data[DataLoader.LABEL_COLUMN].append(0)
            with tf.io.gfile.GFile(os.path.join(self.path, 'train_pos.txt'), 'r') as f:
                for line in f:
                    data[DataLoader.DATA_COLUMN].append(line)
                    data[DataLoader.LABEL_COLUMN].append(1)
        df = pd.DataFrame.from_dict(data)
        df = df.sample(frac=1).reset_index(drop=True)

        df_test = pd.read_csv(os.path.join(self.path, 'test_data.txt'), sep='\n', header=None)
        df_test.columns = [DataLoader.DATA_COLUMN]

        regex = re.compile('^[0-9]*,([\\s\\S]*)')
        for i, row in df_test.iterrows():
            df_test.at[i, DataLoader.DATA_COLUMN] = regex.match(df_test.at[i, DataLoader.DATA_COLUMN]).group(1)
        return df, df_test
