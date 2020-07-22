import os
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
from bunch import Bunch
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset, random_split
from transformers import BertTokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizerFast
from utilities.data_loading import (
    generate_bootstrap_dataset,
    load_tweets,
    remove_indices_from_test_tweets,
    save_labels,
    save_tweets_in_test_format,
)


class DataProcessor:
    def __init__(
        self,
        config: Bunch,
        logger: CometLogger,
        testing: bool,
        tokenizer: PreTrainedTokenizerFast = None,
    ):
        self.config = config

        self.logger = logger
        self.testing = testing
        self.tokenizer = tokenizer

    def load_tweets(self, path: str) -> List[str]:
        loaded_tweets = load_tweets(path)
        self.logger.experiment.log_other(
            key="n_tweets_from:" + path, value=len(loaded_tweets)
        )
        return loaded_tweets

    def load_unique_tweets(self, path: str) -> List[str]:
        def _remove_duplicate_tweets(tweets: List[str]) -> List[str]:
            return list(OrderedDict.fromkeys(tweets).keys())

        unique_tweets = _remove_duplicate_tweets(self.load_tweets(path))
        self.logger.experiment.log_other(
            key="n_unique_tweets_from:" + path, value=len(unique_tweets)
        )
        return unique_tweets

    def generate_labels(
        self, n_negative_samples: int, n_positive_samples: int
    ) -> torch.Tensor:
        return torch.cat(
            (
                torch.zeros(n_negative_samples, dtype=torch.int64),
                torch.ones(n_positive_samples, dtype=torch.int64),
            )
        )

    def tokenize_tweets(
        self, tokenizer: Optional[BertTokenizerFast], tweets: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokenizer is not None:
            tokenized_input = tokenizer.batch_encode_plus(
                tweets,
                max_length=self.config.max_tokens_per_tweet,
                pad_to_max_length=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
            return tokenized_input["input_ids"], tokenized_input["attention_mask"]
        else:
            raise ValueError("Tokenizer is None")

    def train_validation_split(
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

    def prepare_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        negative_tweets = self.load_unique_tweets(self.config.negative_tweets_path)
        positive_tweets = self.load_unique_tweets(self.config.positive_tweets_path)
        all_tweets = negative_tweets + positive_tweets
        labels = self.generate_labels(len(negative_tweets), len(positive_tweets))

        train_token_ids, train_attention_mask = self.tokenize_tweets(
            self.tokenizer, all_tweets
        )

        self.train_data, self.validation_data = self.train_validation_split(
            self.config.validation_size,
            TensorDataset(train_token_ids, train_attention_mask, labels),
            self.config.validation_split_random_seed,
        )

        if not self.testing:
            validation_indices = list(self.validation_data.indices)
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

        test_tweets = self.load_tweets(self.config.test_tweets_path)
        test_tweets_index_removed = remove_indices_from_test_tweets(test_tweets)
        test_token_ids, test_attention_mask = self.tokenize_tweets(
            self.tokenizer, test_tweets_index_removed
        )
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)

        if self.config.use_augmented_data:
            additional_positive_tweets = self.load_unique_tweets(
                self.config.additional_positive_tweets_path
            )
            additional_negative_tweets = self.load_unique_tweets(
                self.config.additional_negative_tweets_path
            )
            additional_labels = self.generate_labels(
                len(additional_negative_tweets), len(additional_positive_tweets)
            )
            (
                additional_train_token_ids,
                additional_train_attention_mask,
            ) = self.tokenize_tweets(
                self.tokenizer, additional_negative_tweets + additional_positive_tweets
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
