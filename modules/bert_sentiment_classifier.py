import os
from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from bunch import Bunch
from modules.data_processor import DataProcessor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
from transformers import BertForSequenceClassification, BertTokenizerFast
from utilities.data_loading import save_labels, save_tweets_in_test_format


class BertSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super().__init__()
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(
            config.pretrained_model
        )
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)
        self.data_processor = DataProcessor(config, tokenizer)
        self.loss = CrossEntropyLoss()

    def save_validation_tweets_and_labels(
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

    def prepare_data(self) -> None:
        (
            self.train_data,
            self.validation_data,
            self.test_data,
        ) = self.data_processor.prepare_data(self.logger)

        (
            negative_tweets,
            positive_tweets,
            labels,
        ) = self.data_processor.get_tweets_and_labels(
            self.config.negative_tweets_path, self.config.positive_tweets_path
        )
        all_tweets = negative_tweets + positive_tweets

        if not self.testing:
            self.save_validation_tweets_and_labels(
                all_tweets, labels, self.validation_data
            )

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
