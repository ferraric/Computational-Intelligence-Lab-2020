from typing import Dict, List, Union

import pytorch_lightning as pl
import torch
from bunch import Bunch
from data_processing.data_processor import DataProcessor
from data_processing.pretrained_tokenizer import PreTrainedTokenizer
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizerFast


class BertSentimentClassifier(pl.LightningModule):
    def __init__(self, config: Bunch) -> None:
        super().__init__()
        self.config = config
        self.model = BertForSequenceClassification.from_pretrained(
            config.pretrained_model
        )
        bert_tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)
        tokenizer = PreTrainedTokenizer(
            self.config.max_tokens_per_tweet, bert_tokenizer
        )
        self.data_processor = DataProcessor(config, tokenizer)
        self.loss = CrossEntropyLoss()

    def prepare_data(self) -> None:
        (
            self.train_data,
            self.validation_data,
            self.test_data,
        ) = self.data_processor.prepare_data(self.testing, self.logger)

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
