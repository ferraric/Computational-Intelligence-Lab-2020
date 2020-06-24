from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from bunch import Bunch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from transformers import BertForSequenceClassification, BertTokenizerFast


class BertSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super().__init__()
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(
            config.pretrained_model
        )
        self.loss = CrossEntropyLoss()

    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        def _load_tweets(path: str) -> List[str]:
            with open(path, encoding="utf-8") as f:
                return f.read().splitlines()

        def _generate_labels(
            n_negative_samples: int, n_positive_samples: int
        ) -> torch.Tensor:
            return torch.cat(
                (
                    torch.zeros(n_negative_samples, dtype=torch.int64),
                    torch.ones(n_positive_samples, dtype=torch.int64),
                )
            )

        def _tokenize_tweets(
            tokenizer: BertTokenizerFast, tweets: List[str]
        ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            return token_ids, attention_mask

        def _train_validation_split(
            validation_size: float, data: TensorDataset
        ) -> List[Subset]:
            assert 0 <= validation_size and validation_size <= 1

            n_validation_samples = int(validation_size * len(data))
            n_train_samples = len(data) - n_validation_samples
            return random_split(data, [n_train_samples, n_validation_samples])

        negative_tweets = _load_tweets(self.config.negative_tweets_path)
        positive_tweets = _load_tweets(self.config.positive_tweets_path)
        labels = _generate_labels(len(negative_tweets), len(positive_tweets))
        train_token_ids, train_attention_mask = _tokenize_tweets(
            tokenizer, negative_tweets + positive_tweets
        )
        self.train_data, self.validation_data = _train_validation_split(
            self.config.validation_size,
            TensorDataset(train_token_ids, train_attention_mask, labels),
        )

        test_tweets = _load_tweets(self.config.test_tweets_path)
        test_token_ids, test_attention_mask = _tokenize_tweets(tokenizer, test_tweets)
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)

        max_sequence_length_train = torch.max(
            torch.sum(train_attention_mask, dim=1)
        ).item()
        max_sequence_length_test = torch.max(
            torch.sum(test_attention_mask, dim=1)
        ).item()
        max_sequence_length = max(max_sequence_length_train, max_sequence_length_test)
        # this will only be known at runtime and should be used for setting
        # the max_tokens_per_tweet config property
        self.logger.log_hyperparams({"actual_max_sequence_length": max_sequence_length})

    def forward(
        self, token_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        (logits,) = self.model(token_ids, attention_mask)
        return logits

    def training_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        token_ids, attention_mask, labels = batch
        logits = self.forward(token_ids, attention_mask)
        loss = self.loss(logits, labels)
        return {"loss": loss}

    def validation_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        token_ids, attention_mask, labels = batch
        logits = self.forward(token_ids, attention_mask)
        loss = self.loss(logits, labels)
        accuracy = (logits.argmax(-1) == labels).float().mean()
        return {"loss": loss, "accuracy": accuracy}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        loss = torch.mean(torch.stack([output["loss"] for output in outputs]))
        accuracy = torch.mean(torch.stack([output["accuracy"] for output in outputs]))
        out = {"val_loss": loss, "val_acc": accuracy}
        return {**out, "log": out}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=False,
        )

    def test_data_loader(self) -> None:
        pass

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.config.learning_rate)
