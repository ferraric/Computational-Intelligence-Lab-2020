import inspect
import os
import sys
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
from bunch import Bunch
from comet_ml import Experiment
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
        self.loss = CrossEntropyLoss(reduction="none")

    def prepare_data(self) -> None:
        tokenizer = BertTokenizerFast.from_pretrained(self.config.pretrained_model)

        def _load_tweets_and_labels() -> Tuple[List[str], torch.Tensor]:
            with open(self.config.negative_tweets_path, encoding="utf-8") as f:
                text_lines_neg = f.read().splitlines()[:100]
            with open(self.config.positive_tweets_path, encoding="utf-8") as f:
                text_lines_pos = f.read().splitlines()[:100]
            tweets = text_lines_neg + text_lines_pos
            labels = torch.cat(
                (
                    torch.zeros(len(text_lines_neg), dtype=torch.int64),
                    torch.ones(len(text_lines_pos), dtype=torch.int64),
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
        loss = self.loss(logits, labels).mean()
        return {"loss": loss}

    def validation_step(
        self, batch: List[torch.Tensor], batch_id: int
    ) -> Dict[str, torch.Tensor]:
        token_ids, attention_mask, labels = batch
        logits = self.forward(token_ids, attention_mask)
        loss = self.loss(logits, labels)
        accuracy = (logits.argmax(-1) == labels).float()
        return {"loss": loss, "accuracy": accuracy}

    def validation_epoch_end(
        self, outputs: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        loss = torch.cat([o["loss"] for o in outputs], 0).mean()
        accuracy = torch.cat([o["accuracy"] for o in outputs], 0).mean()
        return {"validation_loss": loss, "validation_acc": accuracy}

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


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.log_parameters(config)

    model = BertSentimentClassifier(config)
    trainer = pl.Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    main()
