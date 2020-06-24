from typing import List, Tuple

import torch
from bert_sentiment_classifier import BertSentimentClassifier
from bunch import Bunch
from torch.utils.data import Subset, TensorDataset, random_split
from transformers import BertTokenizerFast


class BertSentimentClassifierAug(BertSentimentClassifier):
    def __init__(self, config: Bunch) -> None:
        super().__init__(config)

    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        def _load_tweets_and_labels() -> Tuple[List[str], torch.Tensor]:
            with open(self.config.negative_tweets_path, encoding="utf-8") as f:
                text_lines_neg = f.read().splitlines()
            with open(self.config.positive_tweets_path, encoding="utf-8") as f:
                text_lines_pos = f.read().splitlines()
            with open(
                self.config.augmented_negative_tweets_path, encoding="utf-8"
            ) as f:
                text_lines_neg_aug = f.read().splitlines()
            with open(
                self.config.augmented_positive_tweets_path, encoding="utf-8"
            ) as f:
                text_lines_pos_aug = f.read().splitlines()

            tweets_neg = text_lines_neg + text_lines_neg_aug
            tweets_pos = text_lines_pos + text_lines_pos_aug

            tweets = tweets_neg + tweets_pos
            labels = torch.cat(
                (
                    torch.zeros(len(tweets_neg), dtype=torch.int64),
                    torch.ones(len(tweets_pos), dtype=torch.int64),
                )
            )

            return tweets, labels

        def _tokenize_tweets_and_labels(
            tokenizer: BertTokenizerFast, tweets: List[str], labels: torch.Tensor
        ) -> TensorDataset:
            tokenized_input = tokenizer.batch_encode_plus(
                tweets,
                max_length=self.config.max_tokens_per_tweet,
                pad_to_max_length=True,
                return_token_type_ids=False,
            )
            token_ids = torch.tensor(tokenized_input["input_ids"], dtype=torch.int64)
            attention_mask = torch.tensor(
                tokenized_input["attention_mask"], dtype=torch.int64
            )
            return TensorDataset(token_ids, attention_mask, labels)

        def _train_validation_split(
            validation_size: float, data: TensorDataset
        ) -> List[Subset]:
            assert 0 <= validation_size and validation_size <= 1

            n_validation_samples = int(validation_size * len(data))
            n_train_samples = len(data) - n_validation_samples
            return random_split(data, [n_train_samples, n_validation_samples])

        self.train_data, self.validation_data = _train_validation_split(
            self.config.validation_size,
            _tokenize_tweets_and_labels(tokenizer, *_load_tweets_and_labels()),
        )
