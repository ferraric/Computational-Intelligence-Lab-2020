import inspect
import os
import sys
from typing import List, Tuple

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
                text_lines_neg = f.read().splitlines()
            with open(self.config.positive_tweets_path, encoding="utf-8") as f:
                text_lines_pos = f.read().splitlines()
            tweets = text_lines_neg + text_lines_pos
            labels = torch.cat(
                (
                    torch.zeros(len(text_lines_neg), dtype=torch.int),
                    torch.ones(len(text_lines_pos), dtype=torch.int),
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
            input_ids = torch.tensor(tokenized_input["input_ids"], dtype=torch.int)
            attention_mask = torch.tensor(
                tokenized_input["attention_mask"], dtype=torch.int
            )
            return TensorDataset(input_ids, attention_mask, labels)

        def _train_val_split(val_size: float, data: TensorDataset) -> List[Subset]:
            assert 0 <= val_size and val_size <= 1

            n_val_samples = int(val_size * len(data))
            n_train_samples = len(data) - n_val_samples
            return random_split(data, [n_train_samples, n_val_samples])

        self.train_data, self.val_data = _train_val_split(
            self.config.val_size,
            _tokenize_tweets_and_labels(tokenizer, *_load_tweets_and_labels()),
        )

    def forward(self, input_ids: None) -> None:
        pass

    def training_step(self, batch: None, batch_idx: None) -> None:
        pass

    def validation_step(self, batch: None, batch_idx: None) -> None:
        pass

    def validation_epoch_end(self, outputs: None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.config.batch_size,
            drop_last=False,
            shuffle=False,
        )

    def test_data_loader(self) -> None:
        pass

    def configure_optimizers(self) -> Optimizer:
        return Adam(self.parameters(), lr=self.config.lr)


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
