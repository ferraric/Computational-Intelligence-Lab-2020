import inspect
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from bunch import Bunch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split
from transformers import BertForSequenceClassification, BertTokenizerFast
from utilities.general_utilities import get_args, get_bunch_config_from_json

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))  # type: ignore
)
sys.path.insert(0, currentdir)


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

    def test_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        token_ids, attention_mask = batch
        logits = self.forward(token_ids, attention_mask)
        return {"logits": logits}

    def test_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        logits = torch.cat([output["logits"] for output in outputs], 0)

        positive_probabilities = torch.nn.functional.softmax(logits, dim=1)[:, 1]
        predictions = 2 * (logits[:, 1] > logits[:, 0]) - 1
        ids = np.arange(1, logits.shape[0] + 1)
        logit_table = np.hstack((ids.reshape(-1, 1), logits.numpy()))
        prediction_table = np.hstack((ids, predictions.numpy()))
        probabilities_table = np.hstack((ids, positive_probabilities.numpy()))

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
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=False,
        )

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.config.learning_rate)


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)
    pl.seed_everything(config.random_seed)

    current_timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    save_path = os.path.join(
        config.model_save_directory, config.experiment_name, current_timestamp
    )
    os.makedirs(save_path)

    logger = CometLogger(
        save_dir=save_path,
        workspace=config.comet_workspace,
        project_name=config.comet_project_name,
        api_key=config.comet_api_key if config.use_comet_experiments else None,
        experiment_name=config.experiment_name,
    )
    logger.log_hyperparams(config)

    save_model_callback = ModelCheckpoint(
        os.path.join(save_path, "{epoch}-{val_loss:.2f}"), monitor="val_loss"
    )
    number_of_gpus = 1 if torch.cuda.is_available() else 0
    trainer = pl.Trainer(
        checkpoint_callback=save_model_callback,
        deterministic=True,
        fast_dev_run=config.debug,
        gpus=number_of_gpus,
        logger=logger,
        max_epochs=config.epochs,
    )

    if args.test_model_path is None:
        model = BertSentimentClassifier(config)
        trainer.fit(model)
    else:
        model = BertSentimentClassifier.load_from_checkpoint(
            args.test_model_path, config=config
        )
    trainer.test(model=model)


if __name__ == "__main__":
    main()
