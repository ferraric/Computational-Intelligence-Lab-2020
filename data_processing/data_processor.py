import os
from collections import OrderedDict
from typing import List, Tuple

import torch
from bunch import Bunch
from data_processing.data_loading import (
    generate_bootstrap_dataset,
    load_tweets,
    remove_indices_from_test_tweets,
    save_labels,
    save_tweets_in_test_format,
)
from data_processing.tokenizer import Tokenizer
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset, random_split


class DataProcessor:
    def __init__(
        self, config: Bunch, tokenizer: Tokenizer,
    ):
        self.config = config
        self.tokenizer = tokenizer

    def _load_tweets(self, path: str) -> List[str]:
        def _replace_special_tokens(tweet: str) -> str:
            return tweet.replace("<url>", "HTTPURL").replace("<user>", "@USER")

        tweets = load_tweets(path)
        self.logger.experiment.log_other(key="n_tweets_from:" + path, value=len(tweets))
        if self.config.replace_special_tokens:
            return [_replace_special_tokens(tweet) for tweet in tweets]
        else:
            return tweets

    def _load_unique_tweets(self, path: str) -> List[str]:
        def _remove_duplicate_tweets(tweets: List[str]) -> List[str]:
            return list(OrderedDict.fromkeys(tweets).keys())

        unique_tweets = _remove_duplicate_tweets(self._load_tweets(path))
        self.logger.experiment.log_other(
            key="n_unique_tweets_from:" + path, value=len(unique_tweets)
        )
        return unique_tweets

    def _generate_labels(
        self, n_negative_samples: int, n_positive_samples: int
    ) -> torch.Tensor:
        return torch.cat(
            (
                torch.zeros(n_negative_samples, dtype=torch.int64),
                torch.ones(n_positive_samples, dtype=torch.int64),
            )
        )

    def _train_validation_split(
        self, validation_size: float, data: TensorDataset, random_seed: int
    ) -> List[Subset]:
        assert 0 <= validation_size and validation_size <= 1

        n_validation_samples = int(validation_size * len(data))
        n_train_samples = len(data) - n_validation_samples
        random_state_before_split = torch.get_rng_state()
        torch.manual_seed(random_seed)
        [train_data, validation_data] = random_split(
            data, [n_train_samples, n_validation_samples]
        )
        torch.set_rng_state(random_state_before_split)
        return [train_data, validation_data]

    def _get_tweets_and_labels(
        self, negative_tweets_path: str, positive_tweets_path: str
    ) -> Tuple[List[str], torch.Tensor]:
        negative_tweets = self._load_unique_tweets(negative_tweets_path)
        positive_tweets = self._load_unique_tweets(positive_tweets_path)
        all_tweets = negative_tweets + positive_tweets
        labels = self._generate_labels(len(negative_tweets), len(positive_tweets))
        return all_tweets, labels

    def _save_validation_tweets_and_labels(
        self, all_tweets: List[str], labels: torch.Tensor, validation_data: Subset
    ) -> None:
        validation_indices = list(validation_data.indices)
        validation_tweets = [all_tweets[i] for i in validation_indices]
        validation_labels = labels[validation_indices]
        save_tweets_in_test_format(
            validation_tweets,
            os.path.join(self.config.model_save_path, "validation_data.txt"),
        )
        save_labels(
            validation_labels,
            os.path.join(self.config.model_save_path, "validation_labels.txt"),
        )

    def prepare_data(
        self, testing: bool, logger: CometLogger
    ) -> Tuple[Dataset, Subset, Dataset]:
        self.logger = logger

        all_tweets, labels = self._get_tweets_and_labels(
            self.config.negative_tweets_path, self.config.positive_tweets_path
        )

        train_token_ids, train_attention_mask = self.tokenizer.tokenize_tweets(
            all_tweets
        )

        self.train_data, self.validation_data = self._train_validation_split(
            self.config.validation_size,
            TensorDataset(train_token_ids, train_attention_mask, labels),
            self.config.validation_split_random_seed,
        )

        if not testing:
            self._save_validation_tweets_and_labels(
                all_tweets, labels, self.validation_data
            )

        test_tweets = self._load_tweets(self.config.test_tweets_path)
        test_tweets_index_removed = remove_indices_from_test_tweets(test_tweets)
        test_token_ids, test_attention_mask = self.tokenizer.tokenize_tweets(
            test_tweets_index_removed
        )
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)

        if self.config.use_augmented_data:
            additional_negative_tweets = self._load_unique_tweets(
                self.config.additional_negative_tweets_path
            )
            additional_positive_tweets = self._load_unique_tweets(
                self.config.additional_positive_tweets_path
            )
            additional_labels = self._generate_labels(
                len(additional_negative_tweets), len(additional_positive_tweets)
            )
            (
                additional_train_token_ids,
                additional_train_attention_mask,
            ) = self.tokenizer.tokenize_tweets(
                additional_negative_tweets + additional_positive_tweets
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
