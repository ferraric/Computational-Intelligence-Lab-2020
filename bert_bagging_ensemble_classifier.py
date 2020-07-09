import numpy as np
import torch
from bert_sentiment_classifier import BertSentimentClassifier
from bunch import Bunch
from numpy.random._generator import default_rng
from torch.utils.data import TensorDataset
from transformers import BertTokenizerFast


class BertBaggingEnsembleClassifier(BertSentimentClassifier):
    def __init__(self, config: Bunch):
        super().__init__(config)

    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        negative_tweets = self._load_tweets(self.config.negative_tweets_path)
        positive_tweets = self._load_tweets(self.config.positive_tweets_path)

        labels = self._generate_labels(len(negative_tweets), len(positive_tweets))
        original_dataset_list = [negative_tweets + positive_tweets, labels.tolist()]
        original_dataset = np.array(original_dataset_list)

        bootstrap_dataset = default_rng(self.config.random_seed).choice(
            original_dataset, size=original_dataset.shape[0]
        )

        bootstrap_tweets = torch.from_numpy(bootstrap_dataset[:, 0])
        bootstrap_labels = torch.from_numpy(bootstrap_dataset[:, 1])

        train_token_ids, train_attention_mask = self._tokenize_tweets(
            tokenizer, bootstrap_tweets.tolist()
        )

        self.train_data, self.validation_data = self._train_validation_split(
            self.config.validation_size,
            TensorDataset(train_token_ids, train_attention_mask, bootstrap_labels),
        )

        test_tweets = self._load_tweets(self.config.test_tweets_path)
        test_token_ids, test_attention_mask = self._tokenize_tweets(
            tokenizer, test_tweets
        )
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)
