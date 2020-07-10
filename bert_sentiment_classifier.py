from typing import Dict, List, Tuple, Union

import pytorch_lightning as pl
import torch
from bunch import Bunch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from transformers import BertForSequenceClassification, BertTokenizerFast
from utilities.data_loading import (
    generate_bootstrap_dataset,
    load_tweets,
    remove_indices_from_test_tweets,
    save_labels,
    save_tweets_in_test_format,
)


class BertSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super().__init__()
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(
            config.pretrained_model
        )
        self.loss = CrossEntropyLoss()

    def _load_tweets(self, path: str) -> List[str]:
        loaded_tweets = load_tweets(path)
        self.logger.experiment.log_other(
            key="n_tweets_from:" + path, value=len(loaded_tweets)
        )
        return loaded_tweets

    def _load_unique_tweets(self, path: str) -> List[str]:
        def _remove_duplicate_tweets(tweets: List[str]) -> List[str]:
            return list(set(tweets))

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

    def _tokenize_tweets(
        self, tokenizer: BertTokenizerFast, tweets: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_input = tokenizer.batch_encode_plus(
            tweets,
            max_length=self.config.max_tokens_per_tweet,
            pad_to_max_length=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        return tokenized_input["input_ids"], tokenized_input["attention_mask"]

    def _train_validation_split(
        self, validation_size: float, data: TensorDataset
    ) -> List[Subset]:
        assert 0 <= validation_size and validation_size <= 1

        n_validation_samples = int(validation_size * len(data))
        n_train_samples = len(data) - n_validation_samples
        return random_split(data, [n_train_samples, n_validation_samples])

    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        negative_tweets = self._load_unique_tweets(self.config.negative_tweets_path)
        positive_tweets = self._load_unique_tweets(self.config.positive_tweets_path)
        all_tweets = negative_tweets + positive_tweets
        labels = self._generate_labels(len(negative_tweets), len(positive_tweets))

        train_token_ids, train_attention_mask = self._tokenize_tweets(
            tokenizer, all_tweets
        )

        self.train_data, self.validation_data = self._train_validation_split(
            self.config.validation_size,
            TensorDataset(train_token_ids, train_attention_mask, labels),
        )

        validation_indices = list(self.validation_data.indices)
        validation_tweets = [all_tweets[i] for i in validation_indices]
        validation_labels = labels[validation_indices]
        save_tweets_in_test_format(
            validation_tweets, self.config.validation_tweets_save_path
        )
        save_labels(validation_labels, self.config.validation_labels_save_path)

        if self.config.do_bootstrap_sampling:
            self.train_data = generate_bootstrap_dataset(self.train_data)

        test_tweets = self._load_tweets(self.config.test_tweets_path)
        test_tweets_index_removed = remove_indices_from_test_tweets(test_tweets)
        test_token_ids, test_attention_mask = self._tokenize_tweets(
            tokenizer, test_tweets_index_removed
        )
        self.test_data = TensorDataset(test_token_ids, test_attention_mask)

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

    def test_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        token_ids, attention_mask = batch
        logits = self.forward(token_ids, attention_mask)
        return {"logits": logits}

    def test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        logits = torch.cat([output["logits"] for output in outputs], 0).cpu()

        positive_probabilities = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        predictions = 2 * (logits[:, 1] > logits[:, 0]) - 1
        ids = torch.arange(1, logits.shape[0] + 1)
        logit_table = torch.cat((ids.reshape(-1, 1).float(), logits), dim=1).numpy()
        prediction_table = torch.stack((ids, predictions), dim=1).numpy()
        probabilities_table = torch.stack(
            (ids.float(), positive_probabilities), dim=1
        ).numpy()

        self.logger.experiment.log_table(
            filename="test_logits.csv",
            tabular_data=logit_table,
            headers=["Id", "negative", "positive"],
        )
        self.logger.experiment.log_table(
            filename="test_probabilities.csv",
            tabular_data=probabilities_table,
            headers=["Id", "positive_prob"],
        )
        self.logger.experiment.log_table(
            filename="test_predictions.csv",
            tabular_data=prediction_table,
            headers=["Id", "Prediction"],
        )

        return {"predictions": predictions}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=self.config.n_data_loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.config.n_data_loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=self.config.n_data_loader_workers,
        )

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.config.learning_rate)
