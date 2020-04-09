import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from itertools import repeat
import tqdm
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


class TextPreprocessing():
    def __init__(self, tokenizer, max_sentence_length:int):
        self.LABEL = "[CLS]"
        self.PAD = "[PAD]"
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length

    def preprocess_sample(self, text, label):
        assert isinstance(text, str)
        assert isinstance(label, int)
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.max_sentence_length:
            tokens = tokens[:self.max_sentence_length-1]
            ids = self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[self.LABEL]]
        else:
            pad = [self.tokenizer.vocab[self.PAD]] * (self.max_sentence_length-len(tokens)-1)
            ids = self.tokenizer.convert_tokens_to_ids(tokens) + [self.tokenizer.vocab[self.LABEL]] + pad

        return np.array(ids, dtype="int64"), label

    def process_sentence(self, processor, sentence):
        assert isinstance(processor, TextPreprocessing)
        return processor.preprocess_sample(sentence[1]["text"], sentence[1]["label"])


class BertClassifierDataLoader():
    def __init__(self):
        self.num_cores = cpu_count()

    def create_dataloader(self,
                          dataframe:pd.DataFrame,
                          TextProcessor:TextPreprocessing,
                          batch_size:int,
                          shuffle:bool,
                          validation_percentage:float):
        """
        Process the rows of the dataframe i.e. sentences with labels and return a DataLoader.
        :param dataframe:
        :param TextPreprocessor:
        :param batch_size:
        :param shuffle:
        :param valid_percentage:
        :return:
        """
        assert isinstance(TextProcessor, TextPreprocessing)

        with ProcessPoolExecutor(max_workers=self.num_cores) as executor:
            result = list(executor.map(TextProcessor.process_sentence,
                                  repeat(TextProcessor),
                                  dataframe.iterrows(),
                                  chunksize=len(dataframe) // 10))

        features = [r[0] for r in result]
        labels = [r[1] for r in result]

        dataset = TensorDataset(torch.tensor(features, dtype=torch.long),
                                torch.tensor(labels, dtype=torch.long))

        if validation_percentage is not None:
            valid_size = int(validation_percentage * len(dataframe))
            train_size = len(dataframe) - valid_size
            valid_dataset, train_dataset = random_split(dataset,
                                                        [valid_size, train_size])
            valid_loader = DataLoader(valid_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
            train_loader = DataLoader(train_dataset,
                                      batch_size=batch_size,
                                      shuffle=True)
            return train_loader, valid_loader

        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=0,
                                 shuffle=shuffle,
                                 pin_memory=torch.cuda.is_available())
        return data_loader



