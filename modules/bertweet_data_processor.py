from typing import List, Tuple

import torch
from bunch import Bunch
from fairseq.data import Dictionary
from fairseq.data.encoders.fastbpe import fastBPE
from modules.data_processor import DataProcessor
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset
from utilities.data_loading import (
    generate_bootstrap_dataset,
    remove_indices_from_test_tweets,
)


class BertweetDataProcessor(DataProcessor):
    def __init__(
        self, config: Bunch, bpe: fastBPE, vocab: Dictionary,
    ):
        super().__init__(config)
        self.bpe = bpe
        self.vocab = vocab

    def load_tweets(self, path: str) -> List[str]:
        def _replace_special_tokens(tweet: str) -> str:
            return tweet.replace("<url>", "HTTPURL").replace("<user>", "@USER")

        tweets = super().load_tweets(path)
        if self.config.replace_special_tokens:
            return [_replace_special_tokens(tweet) for tweet in tweets]
        else:
            return tweets

    def split_into_tokens(self, tweet: str) -> str:
        return "<s> " + self.bpe.encode(tweet) + " <s>"

    def encode(self, token_string: str) -> List[int]:
        return (
            self.vocab.encode_line(
                token_string, append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )

    def pad(self, token_ids: List[List[int]], max_token_length: int) -> torch.Tensor:
        pad_token_id = self.vocab.pad()
        actual_max_token_length = max(map(len, token_ids))
        assert actual_max_token_length <= max_token_length, (
            "max token length set too small, needs to be at least "
            + str(actual_max_token_length)
        )
        return torch.tensor(
            [
                token_ids_per_tweet
                + [pad_token_id] * (max_token_length - len(token_ids_per_tweet))
                for token_ids_per_tweet in token_ids
            ]
        )

    def generate_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = self.vocab.pad()
        return torch.tensor(token_ids != pad_token_id, dtype=torch.int64)

    def prepare_data(self, logger: CometLogger) -> Tuple[Dataset, Subset, Dataset]:
        self.logger = logger

        negative_tweets, positive_tweets, labels = super().get_tweets_and_labels(
            self.config.negative_tweets_path, self.config.positive_tweets_path
        )
        all_tweets = negative_tweets + positive_tweets

        train_token_id_list = [
            self.encode(self.split_into_tokens(tweet)) for tweet in all_tweets
        ]
        train_token_ids = self.pad(
            train_token_id_list, self.config.max_tokens_per_tweet
        )
        train_attention_mask = self.generate_attention_mask(train_token_ids)
        self.train_data, self.validation_data = super().train_validation_split(
            self.config.validation_size,
            TensorDataset(train_token_ids, train_attention_mask, labels),
            self.config.validation_split_random_seed,
        )

        test_tweets = self.load_tweets(self.config.test_tweets_path)
        test_tweets_index_removed = remove_indices_from_test_tweets(test_tweets)
        test_token_id_list = [
            self.encode(self.split_into_tokens(tweet))
            for tweet in test_tweets_index_removed
        ]
        test_token_ids = self.pad(test_token_id_list, self.config.max_tokens_per_tweet)
        test_attention_mask = self.generate_attention_mask(test_token_ids)
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)

        if self.config.use_augmented_data:
            additional_negative_tweets = self.load_unique_tweets(
                self.config.additional_negative_tweets_path
            )
            additional_positive_tweets = self.load_unique_tweets(
                self.config.additional_positive_tweets_path
            )
            additional_labels = self.generate_labels(
                len(additional_negative_tweets), len(additional_positive_tweets)
            )
            all_additional_tweets = (
                additional_negative_tweets + additional_positive_tweets
            )
            additional_train_token_id_list = [
                self.encode(self.split_into_tokens(tweet))
                for tweet in all_additional_tweets
            ]

            additional_train_token_ids = self.pad(
                additional_train_token_id_list, self.config.max_tokens_per_tweet
            )
            additional_train_attention_mask = self.generate_attention_mask(
                additional_train_token_ids
            )
            additional_train_data = TensorDataset(
                additional_train_token_ids,
                additional_train_attention_mask,
                additional_labels,
            )

            self.train_data = ConcatDataset([self.train_data, additional_train_data])  # type: ignore

        if self.config.do_bootstrap_sampling:
            self.train_data = generate_bootstrap_dataset(self.train_data)

        return self.train_data, self.validation_data, self.test_data
